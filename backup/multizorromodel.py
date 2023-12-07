from enum import Enum
import functools
from functools import wraps

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack
# from einops.layers.torch import Rearrange

from torchmultimodal.utils.common import ModelOutput
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature, \
    ContrastiveLossOutput

# from beartype import beartype
from beartype.typing import Tuple, Optional, Union

from encoders import BioZorroEncoder

# constants

class TokenTypes(Enum):
    SPLICED = 0  # -> AUDIO
    UNSPLICED = 1  # -> VIDEO
    EXPRESSION = 2
    FUSION = 3
    GLOBAL = 4


# functions

def exists(val):
    return val is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor


def pair(t):
    return (t, t) if not isinstance(t, tuple) else t


def cum_mul(it):
    return functools.reduce(lambda x, y: x * y, it, 1)


def divisible_by(numer, denom):
    return (numer % denom) == 0


# decorators

def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


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


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


# attention

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
            attn_mask=None
    ):
        x = self.norm(x)
        kv_x = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(kv_x).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        
        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class BioZorroLayer(nn.Module):
    def __init__(self, dim, dim_head, heads, ff_mult):
        super().__init__()
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.ff = FeedForward(dim=dim, mult=ff_mult)
                 
    def forward(self, batch, zorro_mask=None):
        batch = self.attn(batch, attn_mask=zorro_mask) + batch
        batch = self.ff(batch) + batch
        return batch

@dataclass
class BioZorroPretrainingLossesCollection(ModelOutput):
    contrastive_loss_spliced_unspliced: Optional[Tensor] = None
    contrastive_loss_spliced_expression: Optional[Tensor] = None
    contrastive_loss_unspliced_expression: Optional[Tensor] = None
    fusion_loss_spliced: Optional[Tensor] = None
    fusion_loss_unspliced: Optional[Tensor] = None
    fusion_loss_expression: Optional[Tensor] = None

@dataclass
class BioZorroPretrainingFLOOMLossesCollection(ModelOutput):
    contrastive_loss_spliced_unspliced: Optional[Tensor] = None
    contrastive_loss_spliced_expression: Optional[Tensor] = None
    contrastive_loss_unspliced_expression: Optional[Tensor] = None
    fusion_loss_spliced: Optional[Tensor] = None
    fusion_loss_unspliced: Optional[Tensor] = None
    fusion_loss_expression: Optional[Tensor] = None
    fusion_loss_invariance: Optional[Tensor] = None


@dataclass
class BioZorroPretrainingLossOutput(ModelOutput):
    losses: BioZorroPretrainingLossesCollection = field(default_factory=BioZorroPretrainingLossesCollection)
    spliced: Optional[Tensor] = None
    unspliced: Optional[Tensor] = None
    expression: Optional[Tensor] = None
    fusion: Optional[Tensor] = None
    global_output: Optional[Tensor] = None


class BioZorroPretrainingLoss(nn.Module):
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
        outputs = BioZorroPretrainingLossOutput()
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


@dataclass
class BioZorroPretrainingLossFLOOMOutput(ModelOutput):
    losses: BioZorroPretrainingFLOOMLossesCollection = field(default_factory=BioZorroPretrainingFLOOMLossesCollection)
    spliced: Optional[Tensor] = None
    unspliced: Optional[Tensor] = None
    expression: Optional[Tensor] = None
    fusion: Optional[Tensor] = None
    fusion_floom_spliced: Optional[Tensor] = None
    fusion_floom_unspliced: Optional[Tensor] = None
    fusion_floom_expression: Optional[Tensor] = None
    global_output: Optional[Tensor] = None


class BioZorroPretrainingLossFLOOM(nn.Module):
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
        self.fusion_loss_invariance = nn.MSELoss()

    def forward(
            self,
            pooled_tokens,
            no_loss = False
    ):
        outputs = BioZorroPretrainingLossFLOOMOutput()
        outputs.spliced = pooled_tokens[:, 0, :].squeeze(1)
        outputs.unspliced = pooled_tokens[:, 1, :].squeeze(1)
        outputs.expression = pooled_tokens[:, 2, :].squeeze(1)
        outputs.fusion = pooled_tokens[:, 3, :].squeeze(1)
        outputs.fusion_floom_spliced = pooled_tokens[:, 4, :].squeeze(1)
        outputs.fusion_floom_unspliced = pooled_tokens[:, 5, :].squeeze(1)
        outputs.fusion_floom_expression = pooled_tokens[:, 6, :].squeeze(1)
        outputs.global_output = pooled_tokens[:, 7, :].squeeze(1)

        if no_loss:
            return outputs

        outputs.losses.contrastive_loss_spliced_unspliced = self.contrastive_loss_spliced_unspliced(outputs.unspliced, outputs.unspliced)
        outputs.losses.contrastive_loss_spliced_expression = self.contrastive_loss_spliced_expression(outputs.spliced, outputs.expression)
        outputs.losses.contrastive_loss_unspliced_expression = self.contrastive_loss_unspliced_expression(outputs.unspliced, outputs.expression)
        outputs.losses.fusion_loss_spliced = self.fusion_loss_spliced(outputs.spliced, outputs.fusion)
        outputs.losses.fusion_loss_unspliced = self.fusion_loss_unspliced(outputs.unspliced, outputs.fusion)
        outputs.losses.fusion_loss_expression = self.fusion_loss_expression(outputs.expression,outputs.fusion)
        outputs.losses.fusion_loss_invariance = self.fusion_loss_invariance(outputs.fusion, outputs.fusion_floom_spliced) + \
                                           self.fusion_loss_invariance(outputs.fusion, outputs.fusion_floom_unspliced) + \
                                           self.fusion_loss_invariance(outputs.fusion, outputs.fusion_floom_expression)
        outputs.loss = outputs.losses.contrastive_loss_spliced_unspliced + \
                       outputs.losses.contrastive_loss_spliced_expression + \
                       outputs.losses.contrastive_loss_unspliced_expression + \
                       outputs.losses.fusion_loss_spliced + \
                       outputs.losses.fusion_loss_unspliced + \
                       outputs.losses.fusion_loss_expression + \
                       outputs.losses.fusion_loss_invariance
        return outputs
# main class


class BioZorro(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            #spliced_input_dim,
            #unspliced_input_dim,
            #expression_input_dim,
            dim_head=64,
            heads=8,
            ff_mult=4,
            num_fusion_tokens=16,
            isolate_fusion_tokens=False,
            vocab_size=36602,
            return_token_types: Tuple[TokenTypes] = (TokenTypes.SPLICED, TokenTypes.UNSPLICED,
                                                     TokenTypes.EXPRESSION, TokenTypes.FUSION,
                                                     TokenTypes.GLOBAL),
            loss=BioZorroPretrainingLoss()
    ):
        super().__init__()
        self.loss = loss

        #TODO split off the return token types into a separate module so that it can be reused for fine tuning
        self.max_return_tokens = len(return_token_types)

        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(list(map(lambda t: t.value, return_token_types)))
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent=False)

        self.return_tokens = nn.Parameter(torch.randn(self.max_return_tokens, dim))
        self.attn_pool = Attention(dim=dim, dim_head=dim_head, heads=heads)
        #End TODO

        self.heads = heads # Added
        self.spliced_embedding = BioZorroEncoder(
            num_embeddings = vocab_size, #vocab size
            embedding_dim = dim #spliced_input_dim,
        )
        self.unspliced_embedding = BioZorroEncoder(
            num_embeddings = vocab_size, #vocab size
            embedding_dim = dim #unspliced_input_dim, #Same as layer dim?
        )
        self.expression_embedding = BioZorroEncoder(
            num_embeddings = vocab_size, #vocab size
            embedding_dim = dim #expression_input_dim, #Same as layer dim?
        )
        self.fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim))

        # transformer

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(BioZorroLayer(dim, dim_head, heads, ff_mult))

        self.norm = LayerNorm(dim)

    def forward(
            self,
            *,
            spliced_data: Optional[Tensor] = None,
            spliced_index: Optional[Tensor] = None,
            unspliced_data: Optional[Tensor] = None,
            unspliced_index: Optional[Tensor] = None,
            expression_data: Optional[Tensor] = None,
            expression_index: Optional[Tensor] = None,
#            spliced_attn_mask: Optional[Tensor] = None,
#            unspliced_attn_mask: Optional[Tensor] = None,
#            expression_attn_mask: Optional[Tensor] = None,
            return_token_indices: Optional[Tuple[int]] = None,
            no_loss = False,
            return_final_hidden_state = False #
    ):
        batch, device = spliced_data.shape[0], spliced_data.device
        spliced_tokens = self.spliced_embedding(spliced_index, spliced_data)
        unspliced_tokens = self.unspliced_embedding(unspliced_index, unspliced_data)
        #for i in range(batch):
        #    for j in range(5):
        #        print(f"Last bit of spliced tokens {i},{j},{unspliced_index[i,-5:]}: {unspliced_tokens[i,-5:,j]}")
        #        print(f"Last bit of spliced tokens {i},{j},{spliced_index[i,-5:]}: {spliced_tokens[i,-5:,j]}")
        expression_tokens = self.expression_embedding(expression_index, expression_data)
        fusion_tokens = repeat(self.fusion_tokens, 'n d -> b n d', b=batch)
        #exit()
        # construct all tokens

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

        # construct mask (thus zorro)

        token_types = torch.tensor(list((
            *((TokenTypes.SPLICED.value,) * spliced_tokens.shape[-2]),
            *((TokenTypes.UNSPLICED.value,) * unspliced_tokens.shape[-2]),
            *((TokenTypes.EXPRESSION.value,) * expression_tokens.shape[-2]),
            *((TokenTypes.FUSION.value,) * fusion_tokens.shape[-2]),
        )), device=device, dtype=torch.long)

        token_types_attend_from = rearrange(token_types, 'i -> i 1')
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')

        # the logic goes
        # every modality, including fusion can attend to self

        zorro_mask = token_types_attend_from == token_types_attend_to

        # fusion can attend to everything

        zorro_mask = zorro_mask | (token_types_attend_from == TokenTypes.FUSION.value)

        # Padding tokens mask
        # Using the index data, but should use an input attention mask properly
        # And a custom Padding token via the Dataloader/collator
        #if spliced_attn_mask:
        padding, ps = pack((
            spliced_index != 0,
            unspliced_index != 0,
            expression_index != 0,
            torch.ones(fusion_tokens.shape[0],
                fusion_tokens.shape[1],
                dtype=torch.bool,
                device=fusion_tokens.device)),  #No mask on fusion tokens
            'b *')
        padding_mask = repeat(padding, 'b j -> b i j', i=padding.shape[-1])
        zorro_mask = zorro_mask * padding_mask
        zorro_mask = repeat(zorro_mask, 'b i j -> b h i j', h=self.heads)
        
        # attend and feedforward

        for layer in self.layers:
            tokens = layer(tokens, zorro_mask)

        tokens = self.norm(tokens)

        #For fine tuning just want the last hidden state here
        if return_final_hidden_state:
            return tokens, token_types_attend_to, padding

        # final attention pooling - each modality pool token can only attend to its own tokens

        return_tokens = self.return_tokens
        return_token_types_tensor = self.return_token_types_tensor

        if exists(return_token_indices):
            assert len(set(return_token_indices)) == len(return_token_indices), 'all indices must be unique'
            assert all([indice < self.max_return_tokens for indice in
                        return_token_indices]), 'indices must range from 0 to max_num_return_tokens - 1'

            return_token_indices = torch.tensor(return_token_indices, dtype=torch.long, device=device)

            return_token_types_tensor = return_token_types_tensor[return_token_indices]
            return_tokens = return_tokens[return_token_indices]

        return_tokens = repeat(return_tokens, 'n d -> b n d', b=batch)
        pool_mask = rearrange(return_token_types_tensor, 'i -> i 1') == token_types_attend_to
        # global queries can attend to all tokens
        pool_mask = pool_mask | (rearrange(return_token_types_tensor, 'i -> i 1') == torch.ones_like(
            token_types_attend_to, dtype=torch.long) * TokenTypes.GLOBAL.value)

        #Padding mask to pool mask
        padding_mask = repeat(padding, 'b j -> b i j', i=pool_mask.shape[0])
        pool_mask = pool_mask * padding_mask
        pool_mask = repeat(pool_mask, 'b i j -> b h i j', h=self.heads)

        pooled_tokens = self.attn_pool(return_tokens, context=tokens, attn_mask=pool_mask) + return_tokens

        return self.loss(pooled_tokens, no_loss)


class BioZorroWithLeaveOneOut(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            # spliced_input_dim,
            # unspliced_input_dim,
            # expression_input_dim,
            dim_head=64,
            heads=8,
            ff_mult=4,
            num_fusion_tokens=16,
            isolate_fusion_tokens=True,
            vocab_size=36601,
            return_token_types: Tuple[TokenTypes] = (TokenTypes.SPLICED,
                                                     TokenTypes.UNSPLICED,
                                                     TokenTypes.EXPRESSION,
                                                     TokenTypes.FUSION,
                                                     TokenTypes.FUSION,
                                                     TokenTypes.FUSION,
                                                     TokenTypes.FUSION,
                                                     TokenTypes.GLOBAL),
            loss=BioZorroPretrainingLossFLOOM()
    ):
        super().__init__()
        self.loss = loss
        self.isolate_fusion_tokens = isolate_fusion_tokens
        # TODO split off the return token types into a separate module so that it can be reused for fine tuning
        self.max_return_tokens = len(return_token_types)

        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(list(map(lambda t: t.value, return_token_types)))
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent=False)

        self.return_tokens = nn.Parameter(torch.randn(self.max_return_tokens, dim))
        self.attn_pool = Attention(dim=dim, dim_head=dim_head, heads=heads)
        # End TODO

        self.heads = heads  # Added
        self.spliced_embedding = BioZorroEncoder(
            num_embeddings=vocab_size,  # vocab size
            embedding_dim=dim  # spliced_input_dim,
        )
        self.unspliced_embedding = BioZorroEncoder(
            num_embeddings=vocab_size,  # vocab size
            embedding_dim=dim  # unspliced_input_dim, #Same as layer dim?
        )
        self.expression_embedding = BioZorroEncoder(
            num_embeddings=vocab_size,  # vocab size
            embedding_dim=dim  # expression_input_dim, #Same as layer dim?
        )
        #self.fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim))
        self.num_fusion_tokens = num_fusion_tokens
        self.fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens*4, dim))

        # transformer

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(BioZorroLayer(dim, dim_head, heads, ff_mult))

        self.norm = LayerNorm(dim)

    def forward(
            self,
            *,
            spliced_data: Optional[Tensor] = None,
            spliced_index: Optional[Tensor] = None,
            unspliced_data: Optional[Tensor] = None,
            unspliced_index: Optional[Tensor] = None,
            expression_data: Optional[Tensor] = None,
            expression_index: Optional[Tensor] = None,
            #            spliced_attn_mask: Optional[Tensor] = None,
            #            unspliced_attn_mask: Optional[Tensor] = None,
            #            expression_attn_mask: Optional[Tensor] = None,
            return_token_indices: Optional[Tuple[int]] = None,
            no_loss=False,
            return_final_hidden_state=False  #
    ):
        batch, device = spliced_data.shape[0], spliced_data.device

        spliced_tokens = self.spliced_embedding(spliced_index, spliced_data)
        unspliced_tokens = self.unspliced_embedding(unspliced_index, unspliced_data)
        # for i in range(batch):
        #    for j in range(5):
        #        print(f"Last bit of spliced tokens {i},{j},{unspliced_index[i,-5:]}: {unspliced_tokens[i,-5:,j]}")
        #        print(f"Last bit of spliced tokens {i},{j},{spliced_index[i,-5:]}: {spliced_tokens[i,-5:,j]}")
        expression_tokens = self.expression_embedding(expression_index, expression_data)
        fusion_tokens = repeat(self.fusion_tokens, 'n d -> b n d', b=batch)

        # construct all tokens

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

        # construct mask (thus zorro)

        token_types = torch.tensor(list((
            *((TokenTypes.SPLICED.value,) * spliced_tokens.shape[-2]),
            *((TokenTypes.UNSPLICED.value,) * unspliced_tokens.shape[-2]),
            *((TokenTypes.EXPRESSION.value,) * expression_tokens.shape[-2]),
            *((TokenTypes.FUSION.value,) * fusion_tokens.shape[-2]),
        )), device=device, dtype=torch.long)

        token_types_attend_from = rearrange(token_types, 'i -> i 1')
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')

        # the logic goes
        # every modality, including fusion can attend to self

        zorro_mask = token_types_attend_from == token_types_attend_to

        # fusion can attend to everything
        #zorro_mask = zorro_mask | (token_types_attend_from == TokenTypes.FUSION.value)

        # Fusion leave one out mask
        floom_mask = [token_types != i for i in range(-1, 3)]
        if self.isolate_fusion_tokens:
            for idx, floom in enumerate(floom_mask):
                a = self.num_fusion_tokens *(-4+idx)
                b = self.num_fusion_tokens*(-3+idx)-1
                floom[-self.num_fusion_tokens*4:] = False
                floom[a:b]= True
        floom_mask = repeat(floom_mask, 'i j -> (i i2) j', i2=self.num_fusion_tokens)
        zorro_mask[token_types == TokenTypes.FUSION.value] = floom_mask
        # Padding tokens mask
        # Using the index data, but should use an input attention mask properly
        # And a custom Padding token via the Dataloader/collator
        # if spliced_attn_mask:
        padding, ps = pack((
            spliced_index != 0,
            unspliced_index != 0,
            expression_index != 0,
            torch.ones(fusion_tokens.shape[0],
                       fusion_tokens.shape[1],
                       dtype=torch.bool,
                       device=fusion_tokens.device)),  # No mask on fusion tokens
            'b *')
        padding_mask = repeat(padding, 'b j -> b i j', i=padding.shape[-1])
        zorro_mask = zorro_mask * padding_mask
        zorro_mask = repeat(zorro_mask, 'b i j -> b h i j', h=self.heads)

        # attend and feedforward

        for layer in self.layers:
            tokens = layer(tokens, zorro_mask)

        tokens = self.norm(tokens)

        # For fine tuning just want the last hidden state here
        if return_final_hidden_state:
            return tokens, token_types_attend_to, padding

        # final attention pooling - each modality pool token can only attend to its own tokens

        return_tokens = self.return_tokens
        return_token_types_tensor = self.return_token_types_tensor

        if exists(return_token_indices):
            assert len(set(return_token_indices)) == len(return_token_indices), 'all indices must be unique'
            assert all([indice < self.max_return_tokens for indice in
                        return_token_indices]), 'indices must range from 0 to max_num_return_tokens - 1'

            return_token_indices = torch.tensor(return_token_indices, dtype=torch.long, device=device)

            return_token_types_tensor = return_token_types_tensor[return_token_indices]
            return_tokens = return_tokens[return_token_indices]

        return_tokens = repeat(return_tokens, 'n d -> b n d', b=batch)
        pool_mask = rearrange(return_token_types_tensor, 'i -> i 1') == token_types_attend_to
        # global queries can attend to all tokens
        pool_mask = pool_mask | (rearrange(return_token_types_tensor, 'i -> i 1') == torch.ones_like(
            token_types_attend_to, dtype=torch.long) * TokenTypes.GLOBAL.value)

        # Fusion leave one out mask for pooling tokens with 3 modalities
        floom_pool_mask = torch.block_diag(torch.ones((1, self.num_fusion_tokens)),
                                           torch.ones((1, self.num_fusion_tokens)),
                                           torch.ones((1, self.num_fusion_tokens)),
                                           torch.ones((1, self.num_fusion_tokens))).to(pool_mask.device)
        select_mask = (return_token_types_tensor == TokenTypes.FUSION.value).unsqueeze(1) * \
                      (token_types == TokenTypes.FUSION.value).unsqueeze(0)
        pool_mask[select_mask] = floom_pool_mask.to(torch.bool).flatten()

        # Padding mask to pool mask
        padding_mask = repeat(padding, 'b j -> b i j', i=pool_mask.shape[0])
        pool_mask = pool_mask * padding_mask
        pool_mask = repeat(pool_mask, 'b i j -> b h i j', h=self.heads)

        pooled_tokens = self.attn_pool(return_tokens, context=tokens, attn_mask=pool_mask) + return_tokens

        return self.loss(pooled_tokens, no_loss)