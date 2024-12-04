import sys
import os
import numpy as np
from itertools import chain
import scipy.io

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
            sub=np.loadtxt(path[2]+'subj{0}.txt.preprocessed'.format(i))
            sub=standardise(sub)
            datadic[i]=sub
        return datadic    
    
    elif dataset=='BIO':
        return {file.split('_')[1].split('-')[1]: standardise(np.loadtxt(path[1]+file)) for i,file in enumerate(os.listdir(path[1])) if file.split('_')[0]=='ts' }
    
    elif dataset=='Mastrandrea':
        datadic={}
        for i in chain(range(1,22),range(23,24),range(25,29),range(30,32), range(34,41)):
            sub=np.loadtxt(path[3]+'subj{0}timeseries.txt'.format(i))
            sub=standardise(sub)
            datadic[i]=sub
        return datadic
    
    elif dataset=='Tommaso':
        return {i:standardise(scipy.io.loadmat(path[4]+'/sub{0}_EPI_HCaal_timeseries'.format(i))['aal_project_tot']) 
                for i in chain(range(1,33),range(34,66))}

    else:
        print ('please insert valid dataset: CamCAN, BIO, Mastrandrea, Tommaso')
