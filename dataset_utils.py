from datasets import load_from_disk

def setup_data(dataset_path, split = 0.1, ds_frac=1.0, ds_seed=42, model = 3):
    ## Dataset processing
    dataset = load_from_disk(dataset_path).with_format('torch')

    #Remove columns depending on number of mods
    if ds_frac < 1.0:
        dataset = dataset.select(list(range(0,int(len(dataset)*ds_frac))))
    if model == 3:
        if 'spliced_counts' in dataset.features.keys():    
            keep = ['expression_index','expression_counts','spliced_index', 'unspliced_index', 'spliced_counts', 'unspliced_counts']
        else:
            keep = ['expression_index','expression_data','spliced_index', 'unspliced_index', 'spliced_data', 'unspliced_data']
    elif model == 2:
        keep = ['spliced_index', 'unspliced_index', 'spliced_counts', 'unspliced_counts']
    elif model == 1:
        keep = ['expression_index', 'expression_counts']
    else:
        raise Exception()
    remove = list()
    for key in dataset.features.keys():
        if key not in keep:
            remove.append(key)
    dataset = dataset.remove_columns(remove)

    #Rename some columns for compatibility
    for val in keep:
        if 'counts' in val:
            dataset = dataset.rename_column(val, val.split('_')[0]+'_data')

    #Do a train test split
    dataset = dataset.train_test_split(split, seed=ds_seed)
    return dataset

def create_attn_masks(sample, columns=['expression_index','spliced_index','unspliced_index'], token=0):
    for column in columns:
        sample[column+'_mask'] = sample[column] == token
    return sample
