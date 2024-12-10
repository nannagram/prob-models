'''Workflow for all subjects'''

import numpy as np
import os
import os.path as op
from prob_models.utils.utils import read_paths, getdata
from grbm import split_train_test, train_rbm


if __name__ == '__main__':
    # Load data 
    paths = read_paths(op.join(os.getcwd(), op.join('..', '..', 'paths.txt')))
    results_dir = op.join(os.getcwd(), 'results')
    datadic = getdata(paths, dataset='CamCAN')

    # Parameters for training RBM
    T, N = np.shape(list(datadic.values())[0])
    tf = 0.8
    n_h = 50
    epochs = 500

    for x in datadic.values():
        X_train, X_test = [], []
        for x in datadic.values():
            _X_train, _X_test = split_train_test(x, train_fraction=tf, 
                                                standardize=True, seed=None)
            X_train.append(_X_train)
            X_test.append(_X_test)
            
        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)

        results, paramas = train_rbm(X_train, X_test, n_hid=[n_h], epochs=epochs)

        # Saving options
        all_results_dir = op.join(results_dir, f'all_subjects')
        if not op.exists(all_results_dir):
            os.makedirs(all_results_dir)
        results.to_csv(op.join(all_results_dir, 'results.csv'))
        np.save(op.join(all_results_dir, 'params.npy'), paramas)

