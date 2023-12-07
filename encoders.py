import torch
from torch import nn
from torch.nn.functional import pad
from torch import Tensor
from typing import Optional
from einops.layers.torch import Rearrange
import functools
import math

def cum_mul(it):
    return functools.reduce(lambda x, y: x * y, it, 1)


class TokenEncoder(nn.Module):
    """
    Just an nn.embedding wrapper
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = 1.0
    ):
        super().__init__()
        self.num_embeddings = num_embeddings #debug
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        return x


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using MLP projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512, padding_value = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value
        self.padding_value = padding_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        pad_mask = x == self.padding_value
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        self.dropout(x)
        x = x.masked_fill(pad_mask, 0.0)
        #print(f"Check c {x.shape}, {x}")
        return x


class TabularEncoder(nn.Module):
    """Tabular encoder"""
    def __init__(self,
                 num_embeddings = 128, #Vocab size
                 embedding_dim = 512, #size of embedding vector
                 padding_idx = 0, #padding (no entry) token
                 dropout = 0.0,
                 max_value = 10000,
                 ):
        super().__init__()
        self.index = torch.arange(num_embeddings).unsqueeze(1)
        self.token_encoder = TokenEncoder(num_embeddings, embedding_dim, padding_idx)
        self.value_encoder = ContinuousValueEncoder(embedding_dim, dropout, max_value, padding_idx)

    def forward(self, batch) -> Tensor:
        x_t = self.token_encoder(self.index)
        x_v = self.value_encoder(batch)
        #x_t = repeat(x_p, 'i -> b i', b = x_v.shape[0]) #May need to use this if it doesn't broadcast
        x = x_t + x_v
        return x


class SparseTabularEncoder(nn.Module):
    """Sparse tabular encoder"""
    def __init__(self,
                 num_embeddings = 36602, #Vocab size
                 embedding_dim = 512, #size of embedding vector
                 padding_idx = 0, #padding (no entry) token
                 dropout = 0.0,
                 max_value = 10000,
                 ):
        super().__init__()
        self.token_encoder = TokenEncoder(num_embeddings, embedding_dim, padding_idx)
        self.value_encoder = ContinuousValueEncoder(embedding_dim, dropout, max_value, padding_idx)

    def forward(self, batch) -> Tensor:
        index = batch['index']
        value = batch['value']
        x_t = self.token_encoder(index)
        x_v = self.value_encoder(value)
        x = x_t + x_v
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        #x = x + self.pe[: x.size(0)]
        #return self.dropout(x)
        return self.dropout(self.pe[: x.size(0)])


class SequenceEncoder(nn.Module):
    """
    Basic sequence encoder with sinusoidal positional embedding
    """
    def __init__(self,
                 num_embeddings = 36602, #Vocab size
                 embedding_dim = 512, #size of embedding vector
                 padding_idx = 0, #padding (no entry) token
                 dropout = 0.0,
                 max_tokens = 1024,
                 ):
        super().__init__()
        self.token_encoder = TokenEncoder(num_embeddings, embedding_dim, padding_idx)
        self.positional_encoder = PositionalEncoder(embedding_dim, dropout, max_tokens)

    def forward(self, batch) -> Tensor:
        x_t = self.token_encoder(batch)
        x_p = self.positional_encoder(batch)
        x = x_t + x_p
        return x


class PatchEncoder(nn.Module):
    """
    Patch encoder for audio spectrograms (a 2D matrix),
    images (a 2d Matrix with 3 channels)
    and video (a 3d tensor with 3 channels)

    # TODO Need to mask this one somehow - based on patches
    """
    def __init__(self,
                 patch_size = (16,16), #2 or 3 dimensions
                 mode="matrix",
                 num_channels=3, #Only for image or video
                 embedding_dim = 512,
                 max_tokens = 1024,
                 dropout: float = 0.1
                 ):
        self.dropout = nn.Dropout(p=dropout)
        assert mode in ["matrix", "image", "video"] #Matrix is for mostly audio spectrograms
        if mode in ["matrix", "image"]:
            assert len(patch_size) == 2
        elif mode in ["video"]:
            assert len(patch_size) == 3
        if mode == "matrix":
            input_dim = cum_mul(self.patch_size)
            rearranger = Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1=patch_size[0], p2=patch_size[1]),
        elif mode == "image":
            input_dim = cum_mul(self.patch_size) * num_channels
            rearranger = Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1=patch_size[0], p2=patch_size[1]),
        elif mode == "video":
            input_dim = cum_mul(self.patch_size) * num_channels
            rearranger = Rearrange('b c (t p1) (h p2) (w p3) -> b t h w (c p1 p2 p3)',
                                   p1=patch_size[0], p2=patch_size[1], p3=patch_size[2])
        self.batch_to_tokens = nn.Sequential(
                rearranger,
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )

        self.index = torch.arange(max_tokens).unsqueeze(1)
        self.embedding = nn.Embedding(max_tokens, embedding_dim) #max_norm?

    def forward(self, batch):
        x_t = self.batch_to_tokens(batch) #Linear projection of each patch
        x_p = self.embedding(self.index) #Learnable positional embedding
        #x_p = repeat(x_p, 'i -> b i', b = x_t.shape[0]) #May need to use this if it doesn't broadcast
        x = x_t + x_p
        return self.dropout(x)

# Dictionaries for accessing Encoders and Collators from config
encoders = {
                "SequenceEncoder": SequenceEncoder,
                "TabularEncoder": TabularEncoder,
                "PatchEncoder": PatchEncoder,
            }

class OldSequenceCollator:
    """
    Sequence collator that also works for sparse tabular data (for it's index vector)
    """
    def __init__(self, pad_token=0, pad_len=2048, attn_mask=False):
        self.pad_token = pad_token
        self.pad_len=pad_len
        self.attn_mask = attn_mask

    def __call__(self, data):
        collated_data = {k: list() for k in data[0].keys()}
        if self.attn_mask:
            for k in [key for key in collated_data.keys() if 'index' in key]:
                collated_data[k.split('_')[0] + '_mask'] = list()
        for d in data:
            for k,v in d.items():
                length = v.shape[-1]
                padded_v = pad(v, (0,self.pad_len-length), mode='constant', value=self.pad_token)
                collated_data[k].append(padded_v)
                if self.attn_mask and 'index' in k:
                    collated_data[k.split('_')[0]+'_mask'].append((padded_v != self.pad_token).to(torch.bool))
        for k,v in collated_data.items():
            collated_data[k]=torch.stack(v)
        return collated_data


class SequenceCollator:
    """
    Sequence collator that also works for dense and sparse tabular data
    For sequence data, input {'index':Tensor}
    For dense tabular, input {'index':Tensor, 'data': Tensor} and set pad_len == table length
    For sparse tabular, input {'index':Tensor, 'data': Tensor} and set pad_len == padded length
    TODO: add truncation
    """
    def __init__(self, pad_token=0, pad_len=2048, attn_mask=True):
        self.pad_token = pad_token
        self.pad_len = pad_len
        self.attn_mask = attn_mask

    def __call__(self, data):
        collated_data = {
            'index': [pad(index, (0, self.pad_len - index.shape[-1]), mode='constant', value=self.pad_token)
                      for index in data['index']]}
        if self.attn_mask:
            collated_data['attention_mask'] = [(padded_index == self.pad_token).to(torch.long) for padded_index in padded_indices]
        if 'value' in data.keys():
            collated_data['value'] = [pad(data, (0, self.pad_len-data.shape[-1]), mode='constant', value=0.0)
                            for data in data['value']]
        return {k: torch.stack(v) for k,v in collated_data.items()}


class PatchCollator:
    """
    Sequence collator that also works for sparse tabular data (for it's index vector)
    data is a dict of {'index': Tensor, 'mask': Tensor}
    """
    def __init__(self, pad_token=0, pad_len=2048, attn_mask=False):
        self.pad_token = pad_token
        self.pad_len = pad_len
        self.attn_mask = attn_mask

    def __call__(self, data):
        collated_data = {k: list() for k in data[0].keys()}
        if self.attn_mask:
            for k in [key for key in collated_data.keys() if 'index' in key]:
                collated_data[k.split('_')[0] + '_mask'] = list()
        for d in data:
            for k,v in d.items():
                length = v.shape[-1]
                padded_v = pad(v, (0,self.pad_len-length), mode='constant', value=self.pad_token)
                collated_data[k].append(padded_v)
                if self.attn_mask and 'index' in k:
                    collated_data[k.split('_')[0]+'_mask'].append((padded_v != self.pad_token).to(torch.bool))
        for k,v in collated_data.items():
            collated_data[k] = torch.stack(v)
        return collated_data


class OldSequenceCollatorWithTargets:
    """
    Create a vector with zeros for targets, instead of treating them as a padded sequence.
    """
    def __init__(self, pad_token=0,
            pad_len = 2048,
            target_name = "velocity",
            target_size=2000,
            target_ids=None,
            norm=[1.0,0.0]):
        self.pad_token = pad_token
        self.pad_len = pad_len
        self.target_name = target_name
        self.norm=norm
        if target_ids:
            self.target_size = len(target_ids)
        else:
            self.target_size = target_size
        self.target_ids = target_ids

    def __call__(self, data):  # (2)
        collated_data = {k: list() for k in data[0].keys() if self.target_name not in k}
        collated_data[self.target_name]=list()
        for d in data:
            for k, v in d.items():
                if self.target_name not in k:
                    length = v.shape[-1]
                    padded_v = pad(v, (0, self.pad_len - length), mode='constant', value=self.pad_token)
                    collated_data[k].append(padded_v)
            #Target data (like velocity)
            targets = torch.zeros(self.target_size, dtype = d[self.target_name+'_data'].dtype)
            if self.target_ids:
                # Find target ids in the set of target indices, and then add the data to the targets
                # In the order specified in target_ids. Must be a better way of doing this (pytorchonic way?)
                for target_idx,target_id in enumerate(self.target_ids):
                    idx = d[self.target_name+'_index'] == target_id
                    x = d[self.target_name+'_data'][idx]
                    y = x.item() if x.nelement() != 0 else 0.0
                    targets[target_idx] = y
            else:
                targets[d[self.target_name+'_index']] = d[self.target_name+'_data']
            targets = targets*self.norm[0]+self.norm[1]
            collated_data[self.target_name].append(targets)
        for k, v in collated_data.items():
            collated_data[k] = torch.stack(v)
        return collated_data


class PatchCollator:
    """
    Sequence collator that also works for sparse tabular data (for it's index vector)
    data is a dict of {'index': Tensor, 'mask': Tensor}
    """
    def __init__(self, pad_token=0, pad_len=2048, attn_mask=False):
        self.pad_token = pad_token
        self.pad_len = pad_len
        self.attn_mask = attn_mask

    def __call__(self, data):
        collated_data = {k: list() for k in data[0].keys()}
        if self.attn_mask:
            for k in [key for key in collated_data.keys() if 'index' in key]:
                collated_data[k.split('_')[0] + '_mask'] = list()
        for d in data:
            for k,v in d.items():
                length = v.shape[-1]
                padded_v = pad(v, (0,self.pad_len-length), mode='constant', value=self.pad_token)
                collated_data[k].append(padded_v)
                if self.attn_mask and 'index' in k:
                    collated_data[k.split('_')[0]+'_mask'].append((padded_v != self.pad_token).to(torch.bool))
        for k,v in collated_data.items():
            collated_data[k] = torch.stack(v)
        return collated_data

collators = {
                #"MultimodalCollator": MultimodalCollator,
                "SequenceCollator": SequenceCollator,
                #"SequenceCollatorWithTargets": SequenceCollatorWithTargets,
                "PatchCollator": PatchCollator
            }
class MultimodalCollator:
    """
    Configurable collator for multimodal data with missing modalities
    Could be used to mask modalities
    """
    def __init__(self, modality_config):
        self.modality_collators = {modality_name: collators[config['type']](**config)
                                   for modality_name,config in modality_config.items()}

    def __call__(self, batch):
        assert self.modality_collators.keys() == batch.keys()
        return {k :self.modality_collators[k](v) for k,v in batch.items()}


