import sys
import os
import numpy as np

def read_add_paths(path):
    with open(path, 'r') as f:
        paths = f.read().splitlines()
    for path in paths:
        sys.path.append(path)
    return paths

def read_paths(path):
    with open(path, 'r') as f:
        paths = f.read().splitlines()
    return paths

def standardise(sub):
    return (sub - np.mean(sub,axis=0)) / np.std(sub,axis=0)

def getdata(path,dataset='CamCAN'):
    if dataset=='CamCAN':
        datadic={}
        for i in range(1,652):
            sub=np.loadtxt(path[-1]+'subj{0}.txt.preprocessed'.format(i))
            sub=standardise(sub)
            datadic[i]=sub
        return datadic    
    elif dataset=='BIO':
        return {file.split('_')[1].split('-')[1]: standardise(np.loadtxt(path[-2]+file)) for i,file in enumerate(os.listdir(path[-2])) if file.split('_')[0]=='ts' }
    else:
        print ('please insert valid dataset: CamCAN, BIO')
