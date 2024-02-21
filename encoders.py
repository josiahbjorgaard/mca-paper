import torch
from torch import nn
from torch.nn.functional import pad
from torch import Tensor
from typing import Optional
from einops import repeat
from einops.layers.torch import Rearrange
import functools
import math
from collections import defaultdict
from utils.dataset import BatchDropout

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
        max_norm: Optional[float] = 1.0,
        **kwargs
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

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512, padding_value = 0.0,  **kwargs):
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
                 **kwargs
                 ):
        super().__init__()
        self.register_buffer('index', torch.arange(num_embeddings)) #TODO Attention mask
        self.token_encoder = TokenEncoder(num_embeddings, embedding_dim, padding_idx)
        self.value_encoder = ContinuousValueEncoder(embedding_dim, dropout, max_value, padding_idx)

    def forward(self, batch) -> Tensor:
        x_t = self.token_encoder(self.index)
        x_v = self.value_encoder(batch['values'])
        assert x_v.shape[1] == self.index.shape[0], f"{x_v.shape[1]} - {self.index.shape[0]}"
        x_t = repeat(x_t, 'i j -> b i j', b = x_v.shape[0]) #May need to use this if it doesn't broadcast
        x = x_t + x_v
        return x, batch['attention_mask']



class SparseTabularEncoder(nn.Module):
    """Sparse tabular encoder"""
    def __init__(self,
                 num_embeddings = 36602, #Vocab size
                 embedding_dim = 512, #size of embedding vector
                 padding_idx = 0, #padding (no entry) token
                 dropout = 0.0,
                 max_value = 10000,
                 **kwargs
                 ):
        super().__init__()
        self.token_encoder = TokenEncoder(num_embeddings, embedding_dim, padding_idx)
        self.value_encoder = ContinuousValueEncoder(embedding_dim, dropout, max_value, padding_idx)

    def forward(self, batch) -> Tensor:
        index = batch['indices']
        value = batch['values']
        x_t = self.token_encoder(index)
        x_v = self.value_encoder(value)
        x = x_t + x_v
        return x, batch['attention_mask']


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048,  **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return self.dropout(self.pe[: x.size(1)].repeat(x.size(0), 1, 1))


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
                 **kwargs
                 ):
        super().__init__()
        self.token_encoder = TokenEncoder(num_embeddings, embedding_dim, padding_idx)
        self.positional_encoder = PositionalEncoder(embedding_dim, dropout, max_tokens)

    def forward(self, batch) -> Tensor:
        #print(batch)
        x_t = self.token_encoder(batch['tokens'])
        x_p = self.positional_encoder(batch['tokens'])
        x = x_t + x_p
        return x, batch['attention_mask']


class EmbeddedSequenceEncoder(nn.Module):
    """
    Already Embedded Sequence, so variable length but no token embeddings
    Token embeddings need to be transformed with Linear layer into embedding
    Space size
    Using it for the CMU preembedded dataset
    """
    def __init__(self,
                 input_size = 128, #Vocab size
                 embedding_dim = 512, #size of embedding vector
                 padding_idx = 0, #padding (no entry) token
                 dropout = 0.0,
                 max_tokens = 1024,
                 **kwargs
                 ):
        super().__init__()
        self.input_size = input_size
        self.embedding_dim=embedding_dim
        self.token_encoder = nn.Sequential(
            #nn.BatchNorm1d(input_size),
            nn.LayerNorm([max_tokens,input_size]),
            nn.Linear(input_size, embedding_dim),
            nn.Dropout(dropout),
            )
        self.positional_encoder = PositionalEncoder(embedding_dim, dropout, max_tokens)

    def forward(self, batch) -> Tensor:
        #print(batch)
        to = batch['tokens'].masked_fill(~batch['attention_mask'].unsqueeze(-1).repeat(1,1,self.input_size).to(torch.bool),0.0)#
        x_t = self.token_encoder(to)
        #x_t = torch.zeros(batch['tokens'].shape[:-1], dtype=pred.dtype, device=pred.device)
        #x_t = x_t.unsqueeze(-1).repeat(1,1,self.embedding_dim)
        #x_t[~batch['attention_mask'],:] = pred 
        x_p = self.positional_encoder(batch['tokens'])
        x = x_t + x_p
        return x, batch['attention_mask']


class PatchEncoder(nn.Module):
    """
    Patch encoder for audio spectrograms (a 2D matrix),
    images (a 2d Matrix with 3 channels)
    and video (a 3d tensor with 3 channels)

    # TODO Need to mask this one somehow - based on patches
        Added it here, but it should be somehow in the collator...
    """
    def __init__(self,
                 patch_size = (16,16), #2 or 3 dimensions
                 mode="matrix",
                 num_channels=0, #Only for image or video
                 embedding_dim = 512,
                 max_tokens = 1024,
                 dropout: float = 0.1,
                 attn_mask = True,
                 pad_token = -10000,
                 **kwargs
                 ):
        super().__init__()
        self.attn_mask = attn_mask
        self.pad_token = -10000
        self.dropout = nn.Dropout(p=dropout)
        self.patch_size = patch_size
        assert mode in ["matrix", "image", "video"] #Matrix is for mostly audio spectrograms
        if mode in ["matrix", "image"]:
            assert len(patch_size) == 2
        elif mode in ["video"]:
            assert len(patch_size) == 3
        if mode == "matrix":
            input_dim = cum_mul(self.patch_size)
            opstr = 'b (h p1) (w p2) -> b (h w) (p1 p2)'
            sizes = {f"p{i}":p for i,p in enumerate(patch_size,1)}
            self.layer = Rearrange(opstr,**sizes)
        elif mode == "image":
            input_dim = cum_mul(self.patch_size) * num_channels
            opstr = 'b c (h p1) (w p2) -> b h w (c p1 p2)'
            sizes = {f"p{i}":p for i,p in enumerate(patch_size,1)}
        elif mode == "video":
            input_dim = cum_mul(self.patch_size) * num_channels
            opstr = 'b c (t p1) (h p2) (w p3) -> b t h w (c p1 p2 p3)'
            sizes = {f"p{i}":p for i,p in enumerate(patch_size,1)}
        self.batch_to_tokens = nn.Sequential(
                Rearrange(opstr,**sizes),
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        self.register_buffer('index', torch.arange(max_tokens))
        self.embedding = nn.Embedding(max_tokens, embedding_dim) #max_norm?
    def forward(self, batch):
        x_t = self.batch_to_tokens(batch['values']) #Linear projection of each patch
        assert x_t.shape[1] == self.index.shape[0], f"{x_t.shape[1]} - {self.index.shape[0]}"
        x_p = self.embedding(repeat(self.index, 'l -> b l', b = x_t.shape[0])) #Learnable positional embedding
        x = x_t + x_p
        attention_mask = torch.all(self.layer(batch['values'])==self.pad_token, dim=-1).to(torch.long) if self.attn_mask else None
        return self.dropout(x), attention_mask

# Dictionaries for accessing Encoders and Collators from config
encoders_dict = {
                "SequenceEncoder": SequenceEncoder,
                "TabularEncoder": TabularEncoder,
                "PatchEncoder": PatchEncoder,
                "EmbeddedSequenceEncoder": EmbeddedSequenceEncoder,
            }


class SequenceCollator:
    """
    Sequence collator that also works for dense and sparse tabular data
    For sequence data, input {'index':Tensor}
    For dense tabular, input {'index':Tensor, 'data': Tensor} and set pad_len == table length
    For sparse tabular, input {'index':Tensor, 'data': Tensor} and set pad_len == padded length
    TODO: add truncation
    """
    def __init__(self, pad_token=0, pad_len=2048, data_col_name='indices', attn_mask=True,  **kwargs):
        self.pad_token = pad_token
        self.pad_len = pad_len
        self.attn_mask = attn_mask
        self.data_col_name = data_col_name

    def __call__(self, data):
        #print(data)
        collated_data = {
            self.data_col_name: [pad(index, (0, self.pad_len - index.shape[-1]), mode='constant', value=self.pad_token)
                      for index in data[self.data_col_name]]}
        if self.attn_mask:
            collated_data['attention_mask'] = [(padded_index == self.pad_token).to(torch.long) for padded_index in collated_data[self.data_col_name]]
        if 'values' in data.keys():
            collated_data['values'] = [pad(data, (0, self.pad_len-data.shape[-1]), mode='constant', value=0.0)
                            for data in data['values']]
        return {k: torch.stack(v) for k,v in collated_data.items()}


class EmbeddedSequenceCollator:
    """
    FOR PREEMBEDDED SEQUENCE DATA (no token embedding)
    For dense tabular, input {'data': 3DTensor (batch, index, embedding)} and set pad_len == max length
    TODO: add truncation
    """
    def __init__(self, pad_token=0, pad_len=2048, data_col_name='values', attn_mask=True, truncate=True, **kwargs):
        self.pad_token = pad_token
        self.pad_len = pad_len
        self.attn_mask = attn_mask
        self.data_col_name = data_col_name
        self.truncate = truncate

    def __call__(self, data):
        #print(data)
        if self.truncate:
            data = {self.data_col_name: [index[:self.pad_len] for index in data[self.data_col_name]]}

        collated_data = {
            "tokens": [pad(index, (0,0,0, self.pad_len - index.shape[-2]), mode='constant', value=self.pad_token)
                      for index in data[self.data_col_name]]}
        if self.attn_mask:
            #only need 1D attention mask for each sample
            collated_data['attention_mask'] = [(padded_index[:,0] == self.pad_token).to(torch.long) for
                                               padded_index in collated_data["tokens"]]
        return {k: torch.stack(v) for k,v in collated_data.items()}


class MatrixCollator:
    """
    2D matrix collator
    """
    def __init__(self, pad_token=-10000, pad_len=2048, attn_mask=True, max_channels=0, **kwargs):
        self.pad_token = pad_token
        self.pad_len = pad_len
        self.max_channels = max_channels
        #self.attn_mask = attn_mask

    def __call__(self, data):
        data['values'] = [torch.full((self.max_channels,self.pad_len),self.pad_token, dtype=torch.float) if x is None else x for x in data['values']]
        collated_data = {
            'values': [pad(x, (0,0,0, self.pad_len - x.shape[0]), mode='constant', value=self.pad_token)
                       for x in data['values']]
        }
        if self.max_channels:
            collated_data['values'] = [x[:,:self.max_channels] for x in collated_data['values']]
        return {k: torch.stack(v) for k,v in collated_data.items()}


class OldSequenceCollator:
    """
    Sequence collator that also works for sparse tabular data (for it's index vector)
    """
    def __init__(self, pad_token=0, pad_len=2048, attn_mask=False,  **kwargs):
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


class OldSequenceCollatorWithTargets:
    """
    Create a vector with zeros for targets, instead of treating them as a padded sequence.
    """
    def __init__(self, pad_token=0,
            pad_len = 2048,
            target_name = "velocity",
            target_size=2000,
            target_ids=None,
            norm=[1.0,0.0],
                 **kwargs
                 ):
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


collators = {
    'matrix': MatrixCollator,
    'sequence': SequenceCollator,
    'embedded_sequence': EmbeddedSequenceCollator
}


class MultimodalCollator:
    """
    Configurable collator for multimodal data with missing modalities
    Could be used to mask modalities
    """
    def __init__(self, modality_config, **kwargs):
        self.modality_collators = {modality_name: collators[config['type']](**config)
                                   for modality_name, config in modality_config.items()}
        #self.modality_dropout = {modality_name: BatchDropout(config['padding'], config['dropout']) if config['dropout'] else None
        #                               for modality_name, config in modality_config.items() if not config['predrop']}


    def __call__(self, batch):
        assert self.modality_collators.keys() <= batch[0].keys(), f"{self.modality_collators.keys()} - {batch[0].keys()}"
        d = defaultdict(lambda: defaultdict(list))
        for b in batch:
            for k in self.modality_collators.keys():
                v = b[k]
                for k2, v2 in v.items():
                    d[k][k2].append(v2)
        batch = {k: self.modality_collators[k](v) for k, v in d.items()} #Collate
        #batch = {k: self.modality_dropout[k](v) if self.modality_dropout[k] else v for k, v in batch.items()} #Dropout
        return batch


