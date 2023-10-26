import torch
from torch import nn
from torch.nn.functional import pad
from torch import Tensor
from typing import Optional


class BioZorroCollator:
    def __init__(self, pad_token=0, pad_len=2048):
        self.pad_token = pad_token
        self.pad_len=pad_len

    def __call__(self, data): #(2)
        collated_data = {k: list() for k in data[0].keys()}
        for d in data:
            for k,v in d.items():
                length = v.shape[-1]
                padded_v = pad(v, (0,self.pad_len-length), mode='constant', value=self.pad_token)
                collated_data[k].append(padded_v)
        for k,v in collated_data.items():
            collated_data[k]=torch.stack(v)
        return collated_data


class BioZorroCollatorWithTargets:
    def __init__(self, pad_token=0, 
            pad_len=2048, 
            target_name="velocity", 
            target_size=36601, 
            target_ids=None, 
            norm=[1.0,0.0]):
        self.pad_token = pad_token
        self.pad_len = pad_len
        self.target_name = target_name
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
                    targets[target_idx]=y    
            else:
                targets[d[self.target_name+'_index']] = d[self.target_name+'_data']
            targets = targets/norm[0]+norm[1]
            collated_data[self.target_name].append(targets)
        for k, v in collated_data.items():
            collated_data[k] = torch.stack(v)
        return collated_data


class BioZorroEncoder(nn.Module):
    def __init__(self,
                 num_embeddings = 18817, #Vocab size
                 embedding_dim = 512, #size of embedding vector
                 padding_idx = 0, #padding (no entry) token
                 dropout = 0.0,
                 max_value = 10000,
                 ):
        super().__init__()
        self.gene_encoder = GeneEncoder(num_embeddings, embedding_dim, padding_idx)
        self.counts_encoder = ContinuousValueEncoder(embedding_dim, dropout, max_value, padding_idx)

    def forward(self, index: Tensor, counts: Tensor) -> Tensor:
        x_g = self.gene_encoder(index)
        x_c = self.counts_encoder(counts)
        x = x_g + x_c
        return x

class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = 1.0
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        #print(f"Check g {x[:,-1,-1]}")
        return x


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
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
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
