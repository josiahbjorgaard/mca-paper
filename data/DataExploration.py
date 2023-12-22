import pandas as pd
import os
import numpy as np
import torchaudio.kaldi_io as tkio
from tqdm import tqdm
import datasets
filename = 'text/sum_train/desc.tok.txt'
def loadfile(filename):
    with open(filename, 'rU') as file:
        summ_dict = {}
        for line in file:
            name = line.strip().split(' ')
            summ_dict[name[0]] = np.array(name[1:])
    return summ_dict

summ_dict = loadfile(filename)
trans_en = loadfile('how2-300h-v1/data/train/text.id.en')
trans_pt = loadfile('how2-300h-v1/data/train/text.id.pt')
resnet101=np.load('how2-300h-v1/features/resnext101-action-avgpool-300h/train.npy')

d = dict()
for i in range(1,10):
    gen = tkio.read_mat_ark(f'how2-300h-v1/features/fbank_pitch_181506/raw_fbank_pitch_all_181506.{i}.ark')
    with tqdm(total=19191) as pbar:
        while True:
            try:
                k,v = next(gen)
                d[k]=v
            except StopIteration:
                break
            except Exception as e:
                print(k)
                print(e)
            pbar.update(1)


def dataset_gen():
    for idx,k in enumerate(trans_en.keys()):
        sample = {}
        sample['vid'] = resnet101[idx,:]
        sample['en'] = trans_en.get(k)
        sample['pt'] = trans_pt.get(k)
        sample['sm'] = summ_dict.get(k.split('_')[0])
        sample['aud'] = d.get(k)
        sample['name'] = k
        yield sample
    return


dataset = datasets.Dataset.from_generator(dataset_gen)
dataset.save_to_disk('all_data.dataset')




