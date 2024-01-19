from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack

#from torchmultimodal.utils.common import ModelOutput
#from utils.contrastive_loss_with_temperature import ContrastiveLossWithTemperature
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature

from encoders import encoders_dict

#import torch_xla.core.xla_model as xm


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
            sim = sim.masked_fill(key_padding_mask, -torch.finfo(sim.dtype).max)
        
        attn = sim.softmax(dim=-1)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v) #j is number of toks
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        if return_attn:
            return self.to_out(out), attn
        else:
            return self.to_out(out)


def attn_mask_hack( attn_mask, key_padding_mask, num_heads):
    # This is a hack for a bug in MHA https://github.com/pytorch/pytorch/issues/41508
    # Combines padding mask into attn mask and adds token-wise self attention to prevent NaNs
    batch, n_tok = key_padding_mask.shape[0], key_padding_mask.shape[1]
    device = key_padding_mask.device
    p=key_padding_mask.unsqueeze(1).repeat(1,n_tok,1)
    z=attn_mask.unsqueeze(0).repeat(batch,1,1)
    attn_mask = (z | p) & ~torch.eye(n_tok,n_tok, device=device).to(torch.bool)
    attn_mask = torch.repeat_interleave(attn_mask, num_heads, dim=0)
    return attn_mask

def pool_mask_hack( pool_mask, key_padding_mask, num_heads):
    # This is a hack for a bug in MHA https://github.com/pytorch/pytorch/issues/41508
    # Combines padding mask into pool mask and adds cross attention that's later masked to prevent NaNs
    batch, n_tok = key_padding_mask.shape[0], pool_mask.shape[0]
    device = key_padding_mask.device
    p=key_padding_mask.unsqueeze(1).repeat(1,n_tok,1)
    z=pool_mask.unsqueeze(0).repeat(batch,1,1)
    attn_mask = (z | p)
    fill_masked_later = torch.all(attn_mask,dim=2).unsqueeze(2).repeat(1,1,key_padding_mask.shape[1])
    attn_mask = attn_mask ^ fill_masked_later
    attn_mask = torch.repeat_interleave(attn_mask, num_heads, dim=0)
    return attn_mask

# attention
class MFDOOMLayer(nn.Module):
    def __init__(self, dim, dim_head, heads, ff_mult):
        super().__init__()
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.num_heads = heads
        #self.attn = nn.MultiheadAttention(embed_dim=dim,
        #                                  num_heads=heads,
        #                                  )
        #                                  #bias=False,
        #                                  #batch_first=True)
        self.ff = FeedForward(dim=dim, mult=ff_mult)
        self.norm = LayerNorm(dim)         
    
    def forward(self, batch, attn_mask=None, padding_mask=None):
        batch = self.norm(batch)
        batch = self.attn(batch, attn_mask=attn_mask, key_padding_mask = padding_mask) + batch
        #batch = batch.swapaxes(0,1)
        #batch = self.attn(batch, batch, batch, attn_mask=attn_mask, key_padding_mask=padding_mask, need_weights=False)[0] + batch
        #batch = self.attn(batch, batch, batch, attn_mask=attn_mask_hack(attn_mask, padding_mask, self.num_heads), need_weights=False)[0] + batch
        #batch = batch.swapaxes(0,1)
        batch = self.norm(batch)
        batch = self.ff(batch) + batch
        return batch


class MFDOOMPretrainingLoss(nn.Module):
    """
    Pairwise contrastive loss.
    N.B. each loss function contains an all-gather operation
    on it's inputs during distributed training
    """
    def __init__(
            self,
            modality_names,
            do_mse,
    ):
        super().__init__()
        self.modality_names = modality_names + ['fusion']
        #TODO check if we really need a separate loss for each one? there's a trainable temperature parameter...
        self.do_mse = do_mse
        if self.do_mse:
            print("Doing MSE")
            #self.mse_losses = {modality_name: nn.MSELoss() for modality_name in modality_names}
            self.mse_losses = {modality_name: ContrastiveLossWithTemperature() for modality_name in modality_names}
        self.losses = {frozenset(pair): ContrastiveLossWithTemperature()
                       for pair in combinations(self.modality_names, r=2)}

    def forward(
            self,
            pooled_tokens,
            sample_mask,
            no_loss = False
    ):
        outputs = {modality_name: pooled_tokens[:,i,:].squeeze(1)
                   for i, modality_name in enumerate(self.modality_names)}

        if self.do_mse: #Add extra fusion tokens
            mlen = len(self.modality_names) #First fusion token is all, the rest are modality-specific
            for i, modality_name in enumerate([x for x in self.modality_names if x != 'fusion']):
                assert i+mlen < pooled_tokens.shape[1]
                outputs[f"fusion_{modality_name}"] = pooled_tokens[:,i+mlen,:].squeeze(1)

        outputs['global_output'] = pooled_tokens[:,-1, :].squeeze(1)
        if not no_loss:
            #Need to apply a sample mask for missing modalities
            # don't compute loss for the pair if one of them is missing
            outputs['losses']={}
            for k in self.losses.keys():
                moda, modb = k
                if moda == 'fusion':
                    mask = sample_mask[modb].to(torch.bool)
                elif modb == 'fusion':
                    mask = sample_mask[moda].to(torch.bool)
                else:
                    mask = (sample_mask[moda] * sample_mask[modb]).to(torch.bool)
                this_loss = self.losses[k](outputs[moda],outputs[modb], mask=mask)
                outputs['losses']["_".join(k)] = this_loss

            if self.do_mse:
                for k in self.mse_losses.keys():
                    mask = sample_mask[k]
                    #this_loss = self.mse_losses[k](outputs['fusion'][mask], outputs[f"fusion_{k}"][mask])
                    this_loss = self.mse_losses[k](outputs['fusion'], outputs[f"fusion_{k}"], mask=mask)
                    outputs['losses'][f"mse_fusion_{k}"] = this_loss

            # Zero out NaN losses (batches with all masked samples give NaN) and average
            loss_tensor = torch.cat([x for x in outputs['losses'].values()])
            loss_mask = ~torch.isnan(loss_tensor)
            nl = torch.sum(loss_mask).to(torch.float)
            if nl == 0.0:
                outputs['loss']=0.0
            else:
                outputs['loss'] = torch.sum(loss_tensor[loss_mask])/nl
        return outputs


class MFDOOM(nn.Module):
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
            inverse_doom = False,
            isolated_fusion = True,
            zorro = False,
            #ntokens=1024,
            **kwargs
    ):
        super().__init__()
        print(f"Got kwargs: {kwargs}")
        self.batch_size = batch_size
        self.isolated_fusion = isolated_fusion
        self.inverse_doom = inverse_doom
        self.fusion_token = -1
        self.global_token = -2
        if zorro:
            return_token_types = list(range(len(encoder_configs))) + [self.fusion_token, self.global_token]
        else: #MFDOOM has extra pooled fusion tokens
            num_pool_fusion_tokens = len(encoder_configs) + 1
            return_token_types = list(range(len(encoder_configs))) + \
                                 [self.fusion_token] * num_pool_fusion_tokens + \
                                 [self.global_token]
        self.max_return_tokens = len(return_token_types)
        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(return_token_types)
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent=False)
        self.return_tokens = nn.Parameter(torch.randn(self.max_return_tokens, dim))

        self.attn_pool = Attention(dim=dim, dim_head=dim_head, heads=heads)
        #self.attn_pool = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,) # bias=False, batch_first=True)
        self.heads = heads

        self.return_padding=return_padding
        self.return_logits=return_logits

        self.encoders = nn.ModuleDict({modality_name: encoders_dict[encoder_config['type']](**encoder_config)
                         for modality_name, encoder_config in encoder_configs.items()})
        self.modality_types = list(encoder_configs.keys())
        
        self.loss = MFDOOMPretrainingLoss(self.modality_types, do_mse = not zorro)

        self.num_fusion_tokens = num_fusion_tokens
        
        self.token_dims = [encoder_configs[token_type]['max_tokens'] for token_type in self.modality_types]
        self.fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim))
        self.register_buffer('fusion_mask',torch.zeros(
                num_fusion_tokens,
            ).to(torch.bool))


        # transformer
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(MFDOOMLayer(dim, dim_head, heads, ff_mult))

        self.norm = LayerNorm(dim)

        self.register_buffer('token_types', self.create_token_types_tensor(self.token_dims, num_fusion_tokens))

        attn_mask = self.create_zorro_mask(self.token_types)
        pool_mask = self.create_zorro_pooling_mask(self.token_types, return_token_types_tensor)
        if not zorro:
            attn_mask = self.create_mfdoom_mask(self.token_types, attn_mask)
            pool_mask = self.create_mfdoom_pooling_mask(self.token_types, return_token_types_tensor, pool_mask)
        self.register_buffer('attn_mask', attn_mask)
        self.register_buffer('pool_mask', pool_mask)

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
        zorro_mask = zorro_mask | (token_types_attend_from == self.fusion_token)
        #zorro_mask = repeat(zorro_mask, 'j i -> i j')
        return ~zorro_mask #.to(torch.long)

    def create_zorro_pooling_mask(self, token_types, return_token_types_tensor):
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')
        pool_mask = rearrange(return_token_types_tensor, 'i -> i 1') == token_types_attend_to
        # global queries can attend to all tokens
        pool_mask = pool_mask | (rearrange(return_token_types_tensor, 'i -> i 1') == torch.ones_like(
            token_types_attend_to, dtype=torch.long) * self.global_token)
        return ~pool_mask

    def create_mfdoom_mask(self, token_types, zorro_mask):
        """
        Create the MFDoom mask. Must have num_fusion_tokens//num_modalities == 0
        """
        # Modal fusion for datasets with only one modality mask
        unique_tokens = torch.unique(token_types)
        assert self.num_fusion_tokens % len(unique_tokens) == 0
        nsubtok = int(self.num_fusion_tokens / len(unique_tokens))
        if self.inverse_doom:
            mfdoom_mask = [token_types == i for i in unique_tokens]
        else:
            mfdoom_mask = [token_types != i if i != self.fusion_token else token_types == i for i in unique_tokens]
        if self.isolated_fusion:
            fusion_tokens = token_types == self.fusion_token
            subfusion_tokens = torch.split(fusion_tokens.nonzero(), nsubtok)
            for idx, mfdoom in enumerate(mfdoom_mask):
                mfdoom[fusion_tokens] = True
                mfdoom[subfusion_tokens[idx]] = False
                mfdoom_mask[idx] = mfdoom
        mfdoom_mask = repeat(mfdoom_mask, 'i j -> (i i2) j', i2=nsubtok)
        zorro_mask[token_types == self.fusion_token] = mfdoom_mask
        return zorro_mask

    def create_mfdoom_pooling_mask(self,
                                   token_types,
                                   return_token_types_tensor,
                                   pool_mask):
        unique_tokens = torch.unique(token_types)
        assert self.num_fusion_tokens % len(unique_tokens) == 0
        nsubtok = int(self.num_fusion_tokens / len(unique_tokens))
        fusion_blocks = [torch.ones((1, nsubtok))
                         for _ in range(len(unique_tokens))]
        mfdoom_pool_mask = torch.block_diag(*fusion_blocks)
        select_mask = (return_token_types_tensor == self.fusion_token).unsqueeze(1) * \
                      (token_types == self.fusion_token).unsqueeze(0)
        pool_mask[select_mask] = ~mfdoom_pool_mask.to(torch.bool).flatten()
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
        fusion_tokens = repeat(self.fusion_tokens, 'n d -> b n d', b=batch_size)
        tokens.append(fusion_tokens)
        tokens, ps = pack(tokens , 'b * d')
        fusion_mask = repeat(self.fusion_mask, 'n -> b n', b=batch_size)
        attention_masks.append(fusion_mask)
        padding, ps = pack(attention_masks, 'b *')
        padding = padding.to(torch.bool)
        # attend and feedforward
        for layer in self.layers:
            tokens = layer(tokens, self.attn_mask, padding)
        return_tokens = repeat(self.return_tokens, 'n d -> b n d', b=batch_size)
        tokens = self.norm(tokens)
        pooled_tokens = self.attn_pool(return_tokens, tokens, attn_mask=self.pool_mask, key_padding_mask = padding) + return_tokens
        loss = self.loss(pooled_tokens, modality_sample_mask,  no_loss)
        return loss
