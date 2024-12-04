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

def getdata(path):
    return {file.split('_')[1].split('-')[1]: standardise(np.loadtxt(path+file)) for i,file in enumerate(os.listdir(path)) if file.split('_')[0]=='ts' }