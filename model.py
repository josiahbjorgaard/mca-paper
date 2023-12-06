from enum import Enum

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack

from torchmultimodal.utils.common import ModelOutput
from utils.contrastive_loss_with_temperature import ContrastiveLossWithTemperature

from beartype.typing import Tuple, Optional, Union

from encoders import encoders

import torch_xla.core.xla_model as xm

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

@dataclass
class MFDOOMPretrainingLossesCollection(ModelOutput):
    contrastive_loss_spliced_unspliced: Optional[Tensor] = None
    contrastive_loss_spliced_expression: Optional[Tensor] = None
    contrastive_loss_unspliced_expression: Optional[Tensor] = None
    fusion_loss_spliced: Optional[Tensor] = None
    fusion_loss_unspliced: Optional[Tensor] = None
    fusion_loss_expression: Optional[Tensor] = None


@dataclass
class MFDOOMLossOutput(ModelOutput):
    losses: BioZorroPretrainingLossesCollection = field(default_factory=BioZorroPretrainingLossesCollection)
    spliced: Optional[Tensor] = None
    unspliced: Optional[Tensor] = None
    expression: Optional[Tensor] = None
    fusion: Optional[Tensor] = None
    global_output: Optional[Tensor] = None


class MFDOOMPretrainingLoss(nn.Module):
    """
    Pairwise contrastive loss.
    N.B. each loss function contains an all-gather operation
    on it's inputs during distributed training
    """
    def __init__(
            self,
    ):
        super().__init__()
        self.contrastive_loss_spliced_unspliced = ContrastiveLossWithTemperature()
        self.contrastive_loss_spliced_expression = ContrastiveLossWithTemperature()
        self.contrastive_loss_unspliced_expression = ContrastiveLossWithTemperature()
        self.fusion_loss_spliced = ContrastiveLossWithTemperature()
        self.fusion_loss_unspliced = ContrastiveLossWithTemperature()
        self.fusion_loss_expression = ContrastiveLossWithTemperature()

    def forward(
            self,
            pooled_tokens,
            no_loss = False
    ):
        outputs = MFDOOMPretrainingLossOutput()
        outputs.spliced = pooled_tokens[:, 0, :].squeeze(1)
        outputs.unspliced = pooled_tokens[:, 1, :].squeeze(1)
        outputs.expression = pooled_tokens[:, 2, :].squeeze(1)
        outputs.fusion = pooled_tokens[:, 3, :].squeeze(1)
        outputs.global_output = pooled_tokens[:, 4, :].squeeze(1)
        if no_loss:
            return outputs

        outputs.losses.contrastive_loss_spliced_unspliced = self.contrastive_loss_spliced_unspliced(outputs.unspliced, outputs.unspliced)
        outputs.losses.contrastive_loss_spliced_expression = self.contrastive_loss_spliced_expression(outputs.spliced, outputs.expression)
        outputs.losses.contrastive_loss_unspliced_expression = self.contrastive_loss_unspliced_expression(outputs.unspliced, outputs.expression)
        outputs.losses.fusion_loss_spliced = self.fusion_loss_spliced(outputs.spliced, outputs.fusion)
        outputs.losses.fusion_loss_unspliced = self.fusion_loss_unspliced(outputs.unspliced, outputs.fusion)
        outputs.losses.fusion_loss_expression = self.fusion_loss_expression(outputs.expression,outputs.fusion)
        outputs.loss = outputs.losses.contrastive_loss_spliced_unspliced + \
                       outputs.losses.contrastive_loss_spliced_expression + \
                       outputs.losses.contrastive_loss_unspliced_expression + \
                       outputs.losses.fusion_loss_spliced + \
                       outputs.losses.fusion_loss_unspliced + \
                       outputs.losses.fusion_loss_expression
        return outputs


class MFDOOM(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            dim_head=64,
            heads=8,
            ff_mult=4,
            num_fusion_tokens=16,
            encoder_configs = {},
            #ntokens=1024,
            return_token_types: Tuple[TokenTypes] = (TokenTypes.SPLICED, TokenTypes.UNSPLICED,
                                                     TokenTypes.EXPRESSION, TokenTypes.FUSION,
                                                     TokenTypes.GLOBAL),
            loss=CombinatorialPretrainingLoss()

    ):
        super().__init__()
       
        self.device = xm.xla_device()

        self.loss = loss

        self.max_return_tokens = len(return_token_types)
        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(list(map(lambda t: t.value, return_token_types)), device=self.device)
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent=False)
        self.return_tokens = nn.Parameter(torch.randn(self.max_return_tokens, dim))

        self.attn_pool = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.heads = heads

        self.encoders = {}
        self.encoders = {modality_name: encoders[encoder_config['encoder_class']](**encoder_config)
                         for modality_name, encoder_config in encoder_configs.items()}
        self.token_types = list(encoders.keys())
        self.token_dims = [encoder_configs[token_type]['n_tokens'] for token_type in self.token_types]

        self.num_fusion_tokens = num_fusion_tokens
        self.fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim))
        self.fusion_mask = torch.ones(
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
        zorro_mask = repeat(zorro_mask, 'j i -> i j')
        return ~zorro_mask

    def create_pooling_mask(self, token_types, return_token_types_tensor):
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')
        pool_mask = rearrange(return_token_types_tensor, 'i -> i 1') == token_types_attend_to
        # global queries can attend to all tokens
        pool_mask = pool_mask | (rearrange(return_token_types_tensor, 'i -> i 1') == torch.ones_like(
            token_types_attend_to, dtype=torch.long) * self.global_token)
        return ~pool_mask

    def forward(
            self,
            *,
            batch, #dict like {modality_name: modality_batch} batch is first index of each modality
            no_loss = False,
    ):
        # Concatenate samples and prepare attention masks
        batch, device = spliced_data.shape[0], spliced_data.device
        spliced_tokens = self.spliced_embedding(spliced_index, spliced_data)
        unspliced_tokens = self.unspliced_embedding(unspliced_index, unspliced_data)
        expression_tokens = self.expression_embedding(expression_index, expression_data)
        fusion_tokens = repeat(self.fusion_tokens, 'n d -> b n d', b=batch)
        fusion_mask = repeat(self.fusion_mask, 'n -> b n', b=batch)

        spliced_tokens, \
        unspliced_tokens, \
        expression_tokens, \
        fusion_tokens = map(lambda t: rearrange(t, 'b ... d -> b (...) d'),
                               (spliced_tokens, unspliced_tokens, expression_tokens, fusion_tokens))

        tokens, ps = pack((
            spliced_tokens,
            unspliced_tokens,
            expression_tokens,
            fusion_tokens
        ), 'b * d')

        padding, ps = pack((
                spliced_mask,
                unspliced_mask,
                expression_mask,
                fusion_mask),
                'b *')
        padding = ~padding #The masks are like 1 1 1 0 0 where 1 denotes non-padding

        # Run model
        # attend and feedforward
        for layer in self.layers:
            tokens = layer(tokens, self.zorro_mask, padding)
        tokens = self.norm(tokens)
        # pooling
        return_tokens = repeat(self.return_tokens, 'n d -> b n d', b=batch)
        pooled_tokens = self.attn_pool(return_tokens, tokens, attn_mask=self.pool_mask, key_padding_mask = padding) + return_tokens
        loss = self.loss(pooled_tokens, no_loss)
        return loss
