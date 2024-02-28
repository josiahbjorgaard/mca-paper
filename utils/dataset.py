from datasets import load_from_disk
import torch
import random

class BatchDropout:
    def __init__(self, kvs = {"attention_mask": 1, "tokens": 0}, dropout=0.1):
        """
        Sets values of a batch mode (a dict of tensors) to constant values
        The constant values can be the padding key and attention mask value
        """
        assert len(kvs) > 0
        self.kvs = kvs
        self.dropout = dropout

    def __call__(self, batch_mode):
        print(batch_mode)
        assert self.kvs.keys() == batch_mode.keys(), print(f"Input {self.kvs.keys()} not all in {batch_mode.keys()}")
        nb = [batch_mode[k].shape[0] for k in self.kvs.keys()][0]
        sz = int((nb*self.dropout))
        if self.dropout == 1.0:
            assert sz == nb
        idx = torch.randperm(nb)[:sz]
        #batch_mode[k] = torch.full_like(batch_mode[k])
        for k, v in self.kvs.items():
            batch_mode[k].index_fill_(0, idx, v)
        return batch_mode


class BatchPreDropout:
    def __init__(self, kvs = {"attention_mask": 1, "tokens": 0}, dropout=0.1):
        """
        Sets values of a batch mode (a dict of tensors) to constant values
        The constant values can be the padding key and attention mask value
        """
        assert len(kvs) > 0
        self.kvs = kvs
        self.dropout = dropout

    def drop(self):
        return torch.rand(1) < self.dropout
    
    def __call__(self, batch_mode):
        """Batch mode is a list in this case"""
        assert self.kvs.keys() == batch_mode.keys(), print(f"Input {self.kvs.keys()} not all in {batch_mode.keys()}")
        #batch_mode[k] = torch.full_like(batch_mode[k])
        if self.drop():
            for k, v in self.kvs.items():
                batch_mode[k] = torch.full_like(batch_mode[k], v) if batch_mode[k] is not None else None
        return batch_mode

def batch_predrop(dataset, modality_config):
    print(modality_config)
    modality_dropout = {
        modality_name: BatchPreDropout(config['padding'], config['dropout']) if config['dropout'] else None
        for modality_name, config in modality_config.items() if config['predrop']}

    def drop(batch):
        return {k: modality_dropout[k](v) if k in modality_dropout.keys() else v for k, v in
                 batch.items()}  # Dropout

    return dataset.map(drop, batched=False)


def setup_data(dataset_path, split = 0.1, ds_frac=1.0, ds_seed=42, model = 3, predrop=False, predrop_config=None):
    ## Dataset processing
    dataset = load_from_disk(dataset_path).with_format('torch')
    if ds_frac < 1.0:
        dataset = dataset.select(list(range(0,int(len(dataset)*ds_frac))))

    if predrop:
        print(f"Running preprocessing dropout of modalities using random seed {torch.seed()}")
        dataset = batch_predrop(dataset, predrop_config)
    #Do a train test split
    if split and split != 1.0:
        dataset = dataset.train_test_split(split, seed=ds_seed)
    return dataset
