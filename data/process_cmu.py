import os
import h5py
from collections import defaultdict
import numpy as np
from datasets import Dataset
import datasets
"This requires about 32GB memory"
def data_generator():
    #File path to CMU processed dataset using script in CMU MultimodalSDK library
    data_dir = '/efs-private/CMU-MultimodalSDK/examples/mmdatasdk_examples/full_examples/final_aligned'
    #filename = os.path.join(data_dir,'All Labels.csd')
    fs=dict()
    for x,vx in {"Labels":'All Labels.csd' ,  
              "COVAREP":"COVAREP.csd" , 
              "FACET":'FACET 4.2.csd' ,  
              "OpenFace":"OpenFace_2.csd" , 
              "glove_vectors":"glove_vectors.csd"}.items():
        fs[x] = h5py.File(os.path.join(data_dir,vx),"r")
    for key in fs['Labels'][list(fs["Labels"].keys())[0]]['data'].keys():
        data = defaultdict()
        for x,vx in fs.items():
            data[x] = np.float32(fs[x][list(fs[x].keys())[0]]['data'][key]['features'][:]) #.keys()
        yield data
ds = Dataset.from_generator(data_generator)
#ds=datasets.load_from_disk('/shared/cmu.dataset')
ds = ds.map(lambda x: {k: {'data':v} for k,v in x.items() if k != "Labels"}, num_proc=16)
ds.save_to_disk('cmu.dataset')
