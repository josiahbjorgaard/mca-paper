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
                 padding_idx = -1, #padding (no entry) token
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
        value = batch['data'] #batch['values']
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
            nn.LayerNorm(input_size),
            nn.Linear(input_size, embedding_dim),
            #nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim)
            )
        self.positional_encoder = PositionalEncoder(embedding_dim, dropout, max_tokens)

    def forward(self, batch) -> Tensor:
        if (~batch['tokens'].isfinite()).sum():
            raise Exception(f"Tokens {batch['tokens'][~batch['tokens'].isfinite()]} are not finite")
        to = batch['tokens'].masked_fill(batch['attention_mask'].unsqueeze(-1).repeat(1,1,self.input_size).to(torch.bool),0.0)
        if (~to.isfinite()).sum():
            raise Exception(f"Bad values in tokens {(~to.isfinite()).sum()}")
        
        x_t = self.token_encoder(to)
        
        x_t = x_t.masked_fill(batch['attention_mask'].unsqueeze(-1).repeat(1,1,x_t.shape[-1]).to(torch.bool),0.0) #Fill these, they are maybe getting a gradient somehow
        if (~x_t.isfinite()).sum():
            raise Exception(f"Encoder transform resulted in {(~x_t.isfinite()).sum()} non-finite values")
        x_p = self.positional_encoder(batch['tokens'])
        x = x_t + x_p
        if x.isnan().sum().sum():
            print(f"{x_t.isnan().sum().sum() = }")
            print(f"{x_p.isnan().sum().sum() = }")
            raise Exception("NaN x")
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
                "SparseTabularEncoder": SparseTabularEncoder,
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
    def __init__(self, pad_token=0, pad_len=2048, data_col_name='indices', other_col='data', attn_mask=True,  **kwargs):
        self.pad_token = pad_token
        self.pad_len = pad_len
        self.attn_mask = attn_mask
        self.data_col_name = data_col_name
        self.other_col = other_col
    def __call__(self, data):
        data = {self.data_col_name: [index if index is not None else torch.empty([0]) for index in data[self.data_col_name]]}

        collated_data = {
            self.data_col_name: [pad(index, (0, self.pad_len - index.shape[-1]), mode='constant', value=self.pad_token)
                      for index in data[self.data_col_name]]}
        if self.attn_mask:
            collated_data['attention_mask'] = [(padded_index == self.pad_token).to(torch.long) for padded_index in collated_data[self.data_col_name]]
        if self.other_col in data.keys():
            collated_data[self.other_col] = [pad(index, (0, self.pad_len-index.shape[-1]), mode='constant', value=0.0)
                            for index in data[self.other_col]]
        return {k: torch.stack(v) for k,v in collated_data.items()}


class EmbeddedSequenceCollator:
    """
    FOR PREEMBEDDED SEQUENCE DATA (no token embedding)
    For dense tabular, input {'data': 3DTensor (batch, index, embedding)} and set pad_len == max length
    TODO: add truncation
    """
    def __init__(self, pad_token=-1, fill_value = 0.0, pad_len=2048, embedding_size=512, data_col_name='values', attn_mask=True, truncate=True, clean=True, **kwargs):
        self.pad_token = pad_token
        self.fill_value = fill_value
        self.pad_len = pad_len
        self.attn_mask = attn_mask
        self.data_col_name = data_col_name
        self.truncate = truncate
        self.clean = clean
        self.embedding_size = embedding_size

    def __call__(self, data):
        data = {self.data_col_name: [index if index is not None else torch.empty([0,self.embedding_size]) for index in data[self.data_col_name]]}
        if self.truncate:
            data = {self.data_col_name: [index[:self.pad_len] for index in data[self.data_col_name]]}
        collated_data = {}
        if self.clean:
            data = {self.data_col_name: [index.nan_to_num() for index in data[self.data_col_name]]}
        if self.attn_mask:
            #only need 1D attention mask for each sample
            collated_data['attention_mask'] = [pad(torch.zeros(index.shape[0],device=index.device), (0, self.pad_len - index.shape[0]), mode='constant', value=1).to(torch.bool)
                                                for index in data[self.data_col_name]]
        collated_data["tokens"]= [pad(index, (0,0,0, self.pad_len - index.shape[-2]), mode='constant', value=self.fill_value)
                      for index in data[self.data_col_name]]
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
    def __init__(self, modality_config, labels=None,**kwargs):
        self.modality_collators = {modality_name: collators[config['type']](**config)
                                   for modality_name, config in modality_config.items()}
        #self.modality_dropout = {modality_name: BatchDropout(config['padding'], config['dropout']) if config['dropout'] else None
        #                               for modality_name, config in modality_config.items() if not config['predrop']}
        self.labels=labels

    def __call__(self, batch):
        assert self.modality_collators.keys() <= batch[0].keys(), f"{self.modality_collators.keys()} - {batch[0].keys()}"
        d = defaultdict(lambda: defaultdict(list))
        for b in batch:
            for k in self.modality_collators.keys():
                v = b[k]
                for k2, v2 in v.items():
                    d[k][k2].append(v2)
                    
        batch_out = {k: self.modality_collators[k](v) for k, v in d.items()} #Collate
        #batch = {k: self.modality_dropout[k](v) if self.modality_dropout[k] else v for k, v in batch.items()} #Dropou
        if self.labels:
            for b in batch:
                v = b[self.labels]
                for k2, v2 in v.items():
                    d[self.labels][k2].append(v2)
            batch_out[self.labels] = {k: torch.stack(v) for k,v in d[self.labels].items()}
        return batch_out


