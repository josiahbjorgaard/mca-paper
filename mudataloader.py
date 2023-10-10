import numpy as np
import torch
from scipy.sparse import csr_matrix
import anndata
import muon as mu
import os
from torch.utils.data import DataLoader
import lightning.pytorch as pl

def sparse_csr_to_tensor(csr:csr_matrix):
    """
    Transform scipy csr matrix to pytorch sparse tensor
    """

    values = csr.data
    indices = np.vstack(csr.nonzero())
    shape = csr.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)
    
def sparse_batch_collate(batch:list):
    """
    Collate function to transform anndata csr view to pytorch sparse tensor
    """
    if type(batch[0]['atac'].X) == anndata._core.views.SparseCSRView:
        atac_batch = sparse_csr_to_tensor(np.vstack([x['atac'].X for x in batch]))
    else:
        atac_batch = torch.FloatTensor(np.vstack([x['atac'].X for x in batch]))

    if type(batch[0]['rna'].X) == anndata._core.views.SparseCSRView:
        rna_batch = sparse_csr_to_tensor(np.vstack([x['rna'].X for x in batch]))
    else:
        rna_batch = torch.FloatTensor(np.vstack([x['rna'].X for x in batch]))
    
    if type(batch[0]['prot'].X) == anndata._core.views.SparseCSRView:
        prot_batch = sparse_csr_to_tensor(np.vstack([x['prot'].X for x in batch]))
    else:
        prot_batch = torch.FloatTensor(np.vstack([x['prot'].X for x in batch]))

    return atac_batch, rna_batch, prot_batch


def get_data(h5mu_file, split=0.8):
    mdata = mu.read(h5mu_file)
    samples = mdata.shape[0]
    train_split = mdata[:int(samples*split),:]
    val_split = mdata[int(samples*split):,:]
    return train_split, val_split

class MuDataModule(pl.LightningDataModule):
    def __init__(self, data_file, batch_size=32):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.train_split, self.val_split = get_data(self.data_file)
    def train_dataloader(self):
        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn = sparse_batch_collate,
            )
    def val_dataloader(self):
        return DataLoader(
            self.val_split,
            batch_size=self.batch_size,
            collate_fn = sparse_batch_collate,
            )
