import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from einops import rearrange, repeat, pack, unpack
from itertools import chain, combinations
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature
#from utils.contrastive_loss_with_temperature import ContrastiveLossWithTemperature
from encoders import encoders_dict


def adjusted_powerset(unique_tokens, powers=[2, 3]):
    yield from chain.from_iterable(combinations(unique_tokens, r) for r in powers)

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def exists(val):
    return val is not None

# bias-less layernorm
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# geglu feedforward
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class FeedForward(nn.Module):
    def __init__(self,
                dim, 
                mult=4):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)
    
        self.feedforward = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            nn.Linear(inner_dim, dim, bias=False)
            )
    def forward(self, batch):
        return self.feedforward(batch)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
            self,
            x,
            context=None,
            attn_mask=None,
            key_padding_mask=None,
            return_attn=False
    ):
        kv_x = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(kv_x).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(attn_mask):
            sim = sim.masked_fill(attn_mask, -torch.finfo(sim.dtype).max)
        if exists(key_padding_mask):
            key_padding_mask = repeat(key_padding_mask, "b i -> b h j i", h=self.heads, j=sim.shape[-2])
            torch.save(key_padding_mask,'key_padding_mask.pt')
            sim = sim.masked_fill(key_padding_mask, -torch.finfo(sim.dtype).max)
        
        attn = sim.softmax(dim=-1)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v) #j is number of toks
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        if return_attn:
            return self.to_out(out), attn
        else:
            return self.to_out(out)


# attention
class MCALayer(nn.Module):
    def __init__(self, dim, dim_head, heads, ff_mult):
        super().__init__()
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.num_heads = heads
        self.ff = FeedForward(dim=dim, mult=ff_mult)
        self.norm = LayerNorm(dim)         
    
    def forward(self, batch, attn_mask=None, padding_mask=None):
        batch = self.norm(batch)
        batch = self.attn(batch, attn_mask=attn_mask, key_padding_mask = padding_mask) + batch
        batch = self.norm(batch)
        batch = self.ff(batch) + batch
        return batch



class MCAPretrainingLoss(nn.Module):
    """
    Pairwise contrastive loss.
    N.B. each loss function contains an all-gather operation
    on it's inputs during distributed training
    """
    def __init__(
            self,
            modality_names,
            bimodal_contrastive = False,
            no_fusion = False,
            non_fusion_fcl = False,
            do_fcl = False, #Should be FrozenSet
            fcl_root = None,
            fusion_combos = None,
            masking = True,
            separate_modal_loss_fns = False
    ):
        super().__init__()
        self.non_fusion_fcl = non_fusion_fcl
        self.masking = masking
        self.modality_names = modality_names
        self.do_fcl = do_fcl
        self.no_fusion = no_fusion
        self.fusion_combos = fusion_combos
        fcl_root = fusion_combos[0]
        self.loss_fn = ContrastiveLossWithTemperature()
        self.loss = ContrastiveLossWithTemperature if separate_modal_loss_fns else lambda: self.loss_fn #If not separate return same instance for all
        if self.do_fcl:
            self.fcl_root = frozenset(fcl_root)
            assert fcl_root in fusion_combos, f"{fcl_root} not in {fusion_combos}"
            self.fcl_losses = {fusion_combo: self.loss() for fusion_combo in fusion_combos if fusion_combo != self.fcl_root}
        else:
            self.fcl_root = None
        if no_fusion:
            loss_pairs = list(combinations(self.modality_names, r=2))
        elif bimodal_contrastive: # Contrast all modalities to all modalities
            loss_pairs = list(combinations(self.modality_names + ['fusion'], r=2))
        else: #Contrast each unmimodal token to fusion
            loss_pairs = [(modality_name, 'fusion') for modality_name in self.modality_names]

        self.losses = {frozenset(pair): self.loss()
                       for pair in loss_pairs}
        
        #Assertion check for loss if no separate modal loss functions, due to the convoluted approach above
        if separate_modal_loss_fns:
            for v in self.losses.values():
                assert v is self.loss()

    def forward(
            self,
            pooled_tokens,
            sample_mask,
            no_loss = False
    ):
        outputs = {modality_name: pooled_tokens[:, i, :].squeeze(1)
                   for i, modality_name in enumerate(self.modality_names)}
        if self.do_fcl: #Add extra fusion tokens
            mlen = len(self.modality_names) #Warn this -1 may need to be removed if fusion token is there #The number of unimodal tokens
            for i, fusion_combo in enumerate(self.fusion_combos):
                assert i+mlen < pooled_tokens.shape[1]
                outputs[fusion_combo] = pooled_tokens[:, i+mlen,:].squeeze(1) #Indexed by tuple of the token indices
            if not self.no_fusion:
                outputs['fusion'] = outputs[self.fcl_root] #Copy it twice because it's easier
        elif not self.no_fusion: #Otherwise just one fusion token after unimodal tokens
            outputs['fusion'] = pooled_tokens[:, len(self.modality_names)].squeeze(1)

        if no_loss:
            return outputs
        #Need to apply a sample mask for missing modalities
        # don't compute loss for the pair if one of them is missing
        outputs['losses'] = {}
        for k in self.losses.keys():
            moda, modb = k
            if not self.masking:
                mask = None
            elif moda == 'fusion':
                mask = sample_mask[modb].to(torch.bool)
            elif modb == 'fusion':
                mask = sample_mask[moda].to(torch.bool)
            else:
                mask = (sample_mask[moda] * sample_mask[modb]).to(torch.bool)
            this_loss = self.losses[k](outputs[moda], outputs[modb], mask=mask)
            outputs['losses']["_".join(sorted(k))] = this_loss
        if self.do_fcl:
            for k in self.fcl_losses.keys():
                mask = torch.stack([sample_mask[self.modality_names[i]] for i in k]).sum(dim=0).to(torch.bool) if self.masking else None
                if not self.no_fusion:
                    this_loss = self.fcl_losses[k](outputs['fusion'], outputs[k], mask=mask)
                    outputs['losses'][f"fcl_fusion|{'_'.join(sorted([self.modality_names[i] for i in k]))}"] = this_loss
                if self.non_fusion_fcl:
                    for mod in self.modality_names:
                        mod_mask = (sample_mask[mod]*mask).to(torch.bool) if self.masking else None
                        this_loss = self.fcl_losses[k](outputs[mod], outputs[k], mask=mod_mask)
                        outputs['losses'][f"fcl_{mod}|{'_'.join(sorted([self.modality_names[i] for i in k]))}"] = this_loss
            outputs['fcl_loss'] = torch.stack([torch.nan_to_num(v) for k,v in outputs['losses'].items() if 'fcl' in k]).mean()
            outputs['no-fcl_loss'] = torch.stack([torch.nan_to_num(v) for k,v in outputs['losses'].items() if 'fcl' not in k]).mean()
        # Zero out NaN losses (batches with all masked samples give NaN) and average
        loss_list = [x for x in outputs['losses'].values()]
        loss_tensor = torch.tensor(loss_list)
        loss_mask = ~torch.isnan(loss_tensor)
        nl = torch.sum(loss_mask).to(torch.float)
        if nl == 0.0:
            print(f"Warning, there are no losses calculated")
            outputs['loss'] = sum([torch.nan_to_num(x) for x in loss_list])
        else:
            outputs['loss'] = sum([torch.nan_to_num(x) for x in loss_list])/nl
        return outputs

class MeanTokenProjectionPool(nn.Module):
    def __init__(self,
                 token_types_tensor,
                 in_dim=512,
                 out_dim=512,
                 projection = True):
        super().__init__()
        self.token_types = token_types_tensor #.unsqueeze(1).repeat(1, in_dim)
        if self.token_types is not None:
            t = token_types_tensor.unique().flip(0)
            t[t>=0] = t[t>=0].flip(0) #Rearrange so the order is the same as other pooling (ids > 0 are at tail)
            self.token_types_unique = t.tolist()
            self.num_token_types = len(token_types_tensor.unique())
            self.proj = nn.ModuleList([nn.Linear(in_dim, out_dim) if projection else nn.Identity() for _ in self.token_types_unique])
        else:
            self.proj = [nn.Linear(in_dim, out_dim) if projection else nn.Identity()]
            self.num_token_types = 1
            self.token_types_unique = [1]
        self.out_dim=out_dim
        #Rearrange token type values so negative token ids go after positives

    def forward(self, batch, key_padding_mask, use_all=False, **kwargs):
        # Batch is b x t x d and reduction is on
        # Scatter padded entries into a dummy index and then discard at the end (masked fill part)
        b, t, d = batch.shape
        output = []
        for idx, layer in zip(self.token_types_unique,self.proj): #For each modality
            matching_indices = (self.token_types == idx).to(key_padding_mask.device) if self.token_types else None
            batch_means = []
            for i in range(b):
                if  not self.token_types:
                    x = batch[i,~key_padding_mask[i,:],:]
                else:
                    x = batch[i,matching_indices * ~key_padding_mask[i,:], :]
                if x.shape[0] == 0:
                    x = torch.zeros(x.shape[1], device=x.device)
                else:
                    x = x.mean(dim=0)
                batch_means.append(x.unsqueeze(0))
            batch_means = torch.cat(batch_means, dim=0)
            output.append(layer(batch_means).unsqueeze(1))
        output = torch.cat(output, dim=1)
        if output.isnan().sum().sum():
            nancount = output.isnan().sum().sum()
            raise Exception(f"NaN in output from Mean Pooling {nancount}, could be from any place before this")
        return output

class MCA(nn.Module):
    def __init__(
            self,
            encoder_configs,
            dim,
            depth,
            dim_head=64,
            heads=8,
            ff_mult=4,
            num_fusion_tokens=16,
            batch_size = 8,
            return_padding = False,
            return_logits = False,
            bimodal_contrastive = False,
            non_fusion_fcl = False,
            fcl = False, #Fusion channel loss
            fcl_root = [1,2,3,4,5], #Root channel for Fusion channel loss, almost always all
            fusion_combos = [4,5], #Powers of fusion channels to mask in attention
            zorro = False, #Replicate the ZORRO paper
            no_fusion = False,
            mean_pool = False,
            **kwargs
    ):
        super().__init__()
        print(f"Got kwargs: {kwargs}")
        self.batch_size = batch_size
        self.no_fusion = no_fusion
        self.inverse_doom = None #inverse_doom #Deprecated
        self.fusion_token = -1
        self.global_token = -2
        self.fusion_combos = [frozenset(x) for x in adjusted_powerset(list(range(len(encoder_configs))), fusion_combos)]
        if no_fusion: # No Fusion loss,
            self.fcl_root=None
            num_fusion_tokens = 0
            return_token_types = list(range(len(encoder_configs))) + [self.global_token]
        elif not fcl or zorro: # No Fusion channel loss or it's using Zorro
            return_token_types = list(range(len(encoder_configs))) + [self.fusion_token, self.global_token]
            self.fcl_root=None
        else: #fusion channel loss has has extra pooled fusion tokens
            self.fcl_root = frozenset(fcl_root)
            num_pool_fusion_tokens = len(self.fusion_combos)
            return_token_types = list(range(len(encoder_configs))) + \
                                 [self.fusion_token] * num_pool_fusion_tokens + \
                                 [self.global_token]
        self.max_return_tokens = len(return_token_types)
        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(return_token_types)
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent=False)


        self.heads = heads

        self.return_padding = return_padding
        self.return_logits = return_logits

        self.encoders = nn.ModuleDict({modality_name: encoders_dict[encoder_config['type']](**encoder_config)
                         for modality_name, encoder_config in encoder_configs.items()})
        self.modality_types = list(encoder_configs.keys())
        self.num_fusion_tokens = num_fusion_tokens 
        
        self.token_dims = [encoder_configs[token_type]['max_tokens'] for token_type in self.modality_types]
        self.fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim))
        self.register_buffer('fusion_mask', torch.zeros(
                num_fusion_tokens,
            ).to(torch.bool))


        # transformer
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(MCALayer(dim, dim_head, heads, ff_mult))

        self.norm = LayerNorm(dim)
        self.register_buffer('token_types', self.create_token_types_tensor(self.token_dims, num_fusion_tokens))
        if mean_pool:
            self.return_tokens = None
            self.attn_pool = MeanTokenProjectionPool(self.token_types, in_dim = dim, out_dim=dim, projection=False)
        else:
            self.return_tokens = nn.Parameter(torch.randn(self.max_return_tokens, dim))
            self.attn_pool = Attention(dim=dim, dim_head=dim_head, heads=heads)
        attn_mask = self.create_zorro_mask(self.token_types)
        pool_mask = self.create_zorro_pooling_mask(self.token_types, return_token_types_tensor)
        if not zorro: #Zorro doesn't have modal-incomplete fusion channel attention mask
            attn_mask = self.create_mca_mask(self.token_types, self.fusion_combos, attn_mask)
            if fcl: #Breaksdown fusion channel tokens in the pooling mask too
                pool_mask = self.create_mca_pooling_mask(self.token_types,
                                                        self.fusion_combos,
                                                        return_token_types_tensor,
                                                        pool_mask)
        self.register_buffer('attn_mask', attn_mask)
        self.register_buffer('pool_mask', pool_mask)

        self.loss = MCAPretrainingLoss(self.modality_types,
                                          do_fcl=fcl and not zorro,
                                          fusion_combos=self.fusion_combos,
                                          fcl_root=self.fcl_root,
                                          bimodal_contrastive = bimodal_contrastive,
                                          no_fusion=no_fusion,
                                          non_fusion_fcl=non_fusion_fcl)


    def create_token_types_tensor(self,dim_list, num_fusion_tokens):
        """
        returns e.g. tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, -1])
        """
        return torch.tensor([x for y in [[i] * n
                                for i,n in enumerate(dim_list)]
                                for x in y]  + [self.fusion_token] * num_fusion_tokens,
                                dtype=torch.long)

    def create_zorro_mask(self, token_types):
        token_types_attend_from = rearrange(token_types, 'i -> i 1')
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')
        zorro_mask = token_types_attend_from == token_types_attend_to
        if not self.no_fusion:
            zorro_mask = zorro_mask | (token_types_attend_from == self.fusion_token)
        return ~zorro_mask

    def create_zorro_pooling_mask(self, token_types, return_token_types_tensor):
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')
        pool_mask = rearrange(return_token_types_tensor, 'i -> i 1') == token_types_attend_to
        # global queries can attend to all tokens
        pool_mask = pool_mask | (rearrange(return_token_types_tensor, 'i -> i 1') == torch.ones_like(
            token_types_attend_to, dtype=torch.long) * self.global_token)
        return ~pool_mask

    def create_mca_mask(self,
                           token_types,
                           fusion_combos,
                           zorro_mask,):
        """
        Create the MCA mask. Must have num_fusion_tokens//num_modalities == 0
        """
        num_fusion_tokens = (token_types == self.fusion_token).sum()
        assert num_fusion_tokens % len(
            fusion_combos) == 0, f"Number of fusion tokens {num_fusion_tokens} must be divisible by the number of combinations {len(fusion_combos)}"
        nsubtok = int(num_fusion_tokens / len(fusion_combos))

        mca_mask = [~torch.isin(token_types, torch.Tensor(list(i))) for i in fusion_combos]

        fusion_tokens = token_types == self.fusion_token
        subfusion_tokens = torch.split(fusion_tokens.nonzero(), nsubtok)
        for idx, mca in enumerate(mca_mask):
            mca[fusion_tokens] = True
            mca[subfusion_tokens[idx]] = False
            mca_mask[idx] = mca
        mca_mask = repeat(mca_mask, 'i j -> (i i2) j', i2=nsubtok)
        zorro_mask[token_types == self.fusion_token] = mca_mask
        return zorro_mask

    def create_mca_pooling_mask(self,
                                   token_types,
                                   fusion_combos,
                                   return_token_types_tensor,
                                   pool_mask):
        assert self.num_fusion_tokens % len(
            fusion_combos) == 0, f"Number of fusion tokens {self.num_fusion_tokens} must be divisible by the number of combinations {len(fusion_combos)}"
        nsubtok = int(self.num_fusion_tokens / len(fusion_combos))
        fusion_blocks = [torch.ones((1, nsubtok))
                         for _ in range(len(fusion_combos))]
        mca_pool_mask = torch.block_diag(*fusion_blocks)
        select_mask = (return_token_types_tensor == self.fusion_token).unsqueeze(1) * \
                      (token_types == self.fusion_token).unsqueeze(0)
        pool_mask[select_mask] = ~mca_pool_mask.to(torch.bool).flatten()
        return pool_mask

    def forward(
            self,
            batch, #dict like {modality_name: modality_data_dict} batch is first index of each modality
            no_loss = False,
    ):
        # Concatenate samples and prepare attention masks
        batch_size =  self.batch_size
        tokens, attention_masks = zip(*[self.encoders[modality_name](batch[modality_name])
                   for modality_name in self.modality_types]) # in case order mixed up
        tokens, attention_masks = list(tokens), list(attention_masks)
        modality_sample_mask = {k:(x==0).sum(dim=1)!=0 for k,x in zip(self.modality_types,attention_masks)}
        if not self.no_fusion:
            fusion_tokens = repeat(self.fusion_tokens, 'n d -> b n d', b=batch_size)
            tokens.append(fusion_tokens)
            fusion_mask = repeat(self.fusion_mask, 'n -> b n', b=batch_size)
            attention_masks.append(fusion_mask)
        tokens, ps = pack(tokens , 'b * d')
        padding, ps = pack(attention_masks, 'b *')
        padding = padding.to(torch.bool)
        # attend and feedforward
        for idx,layer in enumerate(self.layers):
            tokens = layer(tokens, self.attn_mask, padding)
        tokens = self.norm(tokens)
        if self.return_tokens is not None: #Standard Zorro/MCA Pooling
            return_tokens = repeat(self.return_tokens, 'n d -> b n d', b=batch_size)
            pooled_tokens = self.attn_pool(return_tokens, tokens, attn_mask=self.pool_mask, key_padding_mask = padding) + return_tokens
        else:
            pooled_tokens = self.attn_pool(tokens, key_padding_mask = padding) #Mean pooling
        loss = self.loss(pooled_tokens, modality_sample_mask,  no_loss=no_loss)
        loss['modality_sample_mask']=modality_sample_mask
        return loss


class EAO(nn.Module):
    def __init__(
            self,
            encoder_configs,
            dim,
            depth,
            dim_head=64,
            heads=8,
            ff_mult=4,
            num_fusion_tokens=16,
            batch_size=8,
            return_padding=False,
            return_logits=False,
            bimodal_contrastive=False,
            non_fusion_fcl=False,
            fcl=False,  # Fusion channel loss
            fcl_root=[1, 2, 3, 4, 5],  # Root channel for Fusion channel loss, almost always all
            fusion_combos=[4, 5],  # Powers of fusion channels to mask in attention
            zorro=False,  # Replicate the ZORRO paper
            no_fusion=True,
            mean_pool=True,
            **kwargs
    ):
        super().__init__()
        print(f"Got kwargs: {kwargs}")
        self.batch_size = batch_size
        self.fusion_combos = [frozenset(x) for x in adjusted_powerset(list(range(len(encoder_configs))), fusion_combos)]
        self.fcl_root = None
        num_fusion_tokens = 0
        self.fusion_token = -1
        return_token_types = list(range(len(encoder_configs)))
        self.max_return_tokens = len(return_token_types)
        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(return_token_types)
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent=False)

        self.heads = heads

        self.return_padding = return_padding
        self.return_logits = return_logits

        self.encoders = nn.ModuleDict({modality_name: encoders_dict[encoder_config['type']](**encoder_config)
                                       for modality_name, encoder_config in encoder_configs.items()})
        self.modality_types = list(encoder_configs.keys())
        self.token_dims = [encoder_configs[token_type]['max_tokens'] for token_type in self.modality_types]

        # transformer
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(MCALayer(dim, dim_head, heads, ff_mult))
        self.norm = LayerNorm(dim)
        self.register_buffer('token_types', self.create_token_types_tensor(self.token_dims, num_fusion_tokens))
        if mean_pool:
            self.return_tokens = None
            self.attn_pool = MeanTokenProjectionPool(None, in_dim=dim, out_dim=dim, projection=False)
        else:
            self.return_tokens = nn.Parameter(torch.randn(self.max_return_tokens, dim))
            self.attn_pool = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.loss = MCAPretrainingLoss(self.modality_types,
                                       do_fcl=fcl and not zorro,
                                       fusion_combos=self.fusion_combos,
                                       fcl_root=self.fcl_root,
                                       bimodal_contrastive=bimodal_contrastive,
                                       no_fusion=no_fusion,
                                       non_fusion_fcl=non_fusion_fcl)

    def create_token_types_tensor(self, dim_list, num_fusion_tokens):
        """
        returns e.g. tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, -1])
        """
        return torch.tensor([x for y in [[i] * n
                                         for i, n in enumerate(dim_list)]
                             for x in y] + [self.fusion_token] * num_fusion_tokens,
                            dtype=torch.long)

    def single_pass(self,
                    tokens,
                    padding):
        batch_size = tokens.shape[0]
        for idx, layer in enumerate(self.layers):
            tokens = layer(tokens, attn_mask = None, padding_mask = padding)
        tokens = self.norm(tokens)
        if self.return_tokens is not None:  # Standard Zorro/MCA Pooling
            return_tokens = repeat(self.return_tokens, 'n d -> b n d', b=batch_size)
            pooled_tokens = self.attn_pool(return_tokens, tokens, attn_mask=self.pool_mask,
                                           key_padding_mask=padding) + return_tokens
        else:
            pooled_tokens = self.attn_pool(tokens, key_padding_mask=padding)  # Mean pooling
        return pooled_tokens

    def forward(
            self,
            batch,  # dict like {modality_name: modality_data_dict} batch is first index of each modality
            no_loss=False,
    ):
        # Concatenate samples and prepare attention masks
        batch_size = self.batch_size
        tokens, attention_masks = zip(*[self.encoders[modality_name](batch[modality_name])
                                        for modality_name in self.modality_types])  # in case order mixed up
        all_tokens, all_attention_masks = list(tokens), list(attention_masks)
        modality_sample_mask = {k: (x == 0).sum(dim=1) != 0 for k, x in zip(self.modality_types, attention_masks)}

        #Run a forward pass of the transformer attention block for each modality or each set of modalities
        pooled_tokens = []
        for token_types in [[x] for x in range(len(self.modality_types))] + self.fusion_combos:
            tokens, ps = pack([all_tokens[i] for i in token_types], 'b * d')
            padding, ps = pack([all_attention_masks[i] for i in token_types], 'b *')
            padding = padding.to(torch.bool)
            pooled_tokens.append(self.single_pass(tokens, padding))
        pooled_tokens = torch.cat(pooled_tokens, dim=1)

        # attend and feedforward
        loss = self.loss(pooled_tokens, modality_sample_mask, no_loss=no_loss)
        loss['modality_sample_mask'] = modality_sample_mask
        
        return loss
