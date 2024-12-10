''' Workflow for single subject '''

import numpy as np
from numpy.random import default_rng
import os
import os.path as op
from prob_models.utils.utils import read_paths, getdata
from grbm import split_train_test, train_rbm

def single_rbm(X, n_hid, tf=0.8, epochs=100):
    X_train, X_test = split_train_test(X, train_fraction=tf, standardize=True, seed=None)
    results, paramas = train_rbm(X_train, X_test, n_hid, epochs)
    return results, paramas

if __name__ == '__main__':
    # Load data 
    paths = read_paths(op.join(os.getcwd(), op.join('..', '..', 'paths.txt')))
    results_dir = op.join(os.getcwd(), 'results')
    datadic = getdata(paths, dataset='CamCAN')
    T, N = np.shape(list(datadic.values())[0])

    tf = 0.8
    n_h = 25


    for sj, x in datadic.items():
        results, params = single_rbm(x, n_hid=[n_h], tf=0.8, epochs=100)
        
        sub_results_dir = op.join(results_dir, f'sub_{sj}')
        if not op.exists(sub_results_dir):
            os.makedirs(sub_results_dir)
        results.to_csv(op.join(sub_results_dir, 'results.csv'))
        np.save(op.join(sub_results_dir, 'params.npy'), params)
