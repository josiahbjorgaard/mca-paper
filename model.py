#from enum import Enum

#from dataclasses import dataclass, field

from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack

#from torchmultimodal.utils.common import ModelOutput
#from utils.contrastive_loss_with_temperature import ContrastiveLossWithTemperature
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature
from beartype.typing import Tuple, Optional, Union

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

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
            self,
            x,
            context=None,
            attn_mask=None,
            key_padding_mask=None,
    ):
        x = self.norm(x)
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

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# attention
class MFDOOMLayer(nn.Module):
    def __init__(self, dim, dim_head, heads, ff_mult):
        super().__init__()
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.ff = FeedForward(dim=dim, mult=ff_mult)
                 
    def forward(self, batch, zorro_mask=None, padding_mask=None):
        batch = self.attn(batch, batch, attn_mask=zorro_mask, key_padding_mask=padding_mask) + batch
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
    ):
        super().__init__()
        self.modality_names = modality_names
        modality_names = self.modality_names + ['fusion']
        #TODO check if we really need a separate loss for each one? there's a trainable temperature parameter...
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
        outputs['global_output'] = pooled_tokens[:,-1, :].squeeze(1)
        if not no_loss:
            #Need to apply a sample mask for missing modalities
            # don't compute loss for the pair if one of them is missing
            outputs['losses']={}
            for k in self.losses.keys():
                moda, modb = k
                mask = sample_mask[moda] * sample_mask[modb]
                if sum(mask)!=0:
                    outputs['losses']["_".join(k)] = self.losses[k](outputs[moda][mask],outputs[modb][mask])
            outputs['loss'] = torch.sum(torch.tensor(list(outputs['losses'].values()))) #Hopefully this works
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
            device = None,
            #ntokens=1024,

    ):
        super().__init__()
       
        self.device = device #xm.xla_device()
        self.batch_size = batch_size

        return_token_types = list(range(len(encoder_configs))) + [-1, -2]
        self.max_return_tokens = len(return_token_types)
        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(return_token_types, device=self.device)
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent=False)
        self.return_tokens = nn.Parameter(torch.randn(self.max_return_tokens, dim))

        self.attn_pool = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.heads = heads

        self.encoders = {modality_name: encoders_dict[encoder_config['type']](**encoder_config)
                         for modality_name, encoder_config in encoder_configs.items()}
        self.modality_types = list(encoder_configs.keys())

        self.loss = MFDOOMPretrainingLoss(self.modality_types)

        self.num_fusion_tokens = num_fusion_tokens
        
        self.token_dims = [encoder_configs[token_type]['max_tokens'] for token_type in self.modality_types]
        self.fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim))
        self.fusion_mask = torch.zeros(
                num_fusion_tokens,
                device=self.device
            ).to(torch.bool)
        self.fusion_token = -1
        self.global_token = -2

        # transformer
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(MFDOOMLayer(dim, dim_head, heads, ff_mult))

        self.norm = LayerNorm(dim)

        self.token_types = self.create_token_types_tensor(self.token_dims, num_fusion_tokens, self.device)
        self.zorro_mask = self.create_zorro_mask(self.token_types)
        self.pool_mask = self.create_pooling_mask(self.token_types, return_token_types_tensor)

    def create_token_types_tensor(self,dim_list, num_fusion_tokens, device):
        """
        returns e.g. tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, -1])
        """
        return torch.tensor([x for y in [[i] * n
                                for i,n in enumerate(dim_list)]
                                for x in y]  + [self.fusion_token] * num_fusion_tokens,
                                device=device, dtype=torch.long)

    def create_zorro_mask(self, token_types):
        token_types_attend_from = rearrange(token_types, 'i -> i 1')
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')
        zorro_mask = token_types_attend_from == token_types_attend_to
        zorro_mask = zorro_mask | (token_types_attend_from == self.fusion_token)
        #zorro_mask = repeat(zorro_mask, 'j i -> i j')
        return ~zorro_mask #.to(torch.long)

    def create_pooling_mask(self, token_types, return_token_types_tensor):
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')
        pool_mask = rearrange(return_token_types_tensor, 'i -> i 1') == token_types_attend_to
        # global queries can attend to all tokens
        pool_mask = pool_mask | (rearrange(return_token_types_tensor, 'i -> i 1') == torch.ones_like(
            token_types_attend_to, dtype=torch.long) * self.global_token)
        return ~pool_mask

    def forward(
            self,
            batch, #dict like {modality_name: modality_data_dict} batch is first index of each modality
            no_loss = False,
    ):
        # Concatenate samples and prepare attention masks
        batch_size, device = self.batch_size, self.device
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
        # attend and feedforward
        for layer in self.layers:
            tokens = layer(tokens, self.zorro_mask, padding)
        tokens = self.norm(tokens)
        # pooling
        return_tokens = repeat(self.return_tokens, 'n d -> b n d', b=batch_size)
        pooled_tokens = self.attn_pool(return_tokens, tokens, attn_mask=self.pool_mask, key_padding_mask = padding) + return_tokens
        loss = self.loss(pooled_tokens, modality_sample_mask,  no_loss)
        return loss
