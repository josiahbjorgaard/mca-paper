import os
import h5py
from collections import defaultdict
import numpy as np
def data_generator():
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
from datasets import Dataset
ds = Dataset.from_generator(data_generator)
ds.save_to_disk('cmu.dataset')
