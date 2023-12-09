from datasets import load_from_disk

def setup_data(dataset_path, split = 0.1, ds_frac=1.0, ds_seed=42, model = 3):
    ## Dataset processing
    dataset = load_from_disk(dataset_path).with_format('torch')
    if ds_frac < 1.0:
        dataset = dataset.select(list(range(0,int(len(dataset)*ds_frac))))
    #Do a train test split
    dataset = dataset.train_test_split(split, seed=ds_seed)
    return dataset
