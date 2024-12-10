'''Nanna workflow'''

import numpy as np
import os
import os.path as op
from prob_models.utils.utils import read_paths, getdata
from pydeep.rbm.sampler import GibbsSampler
from grbm import split_train_test, learn_rbm

def dreams(ndreams,gibbssampler, n_v,k=400):
    dreams=[]
    v = np.zeros(n_v)
    for i in range(ndreams):
        # v=random.sample(test_data,1)
        v=gibbssampler.sample(v, k=k, betas=None, ret_states=True)[0]
        dreams.append(v)
    return np.array(dreams)


if __name__ == '__main__':
    # Load data 
    paths = read_paths(op.join(os.getcwd(), op.join('..', '..', 'paths.txt')))
    results_dir = op.join(os.getcwd(), 'results')
    datadic = getdata(paths, dataset='CamCAN')

    # Extracting subject from dataset
    X1 = datadic[1]
    tf = 0.8 
    X1_train, X1_test = split_train_test(X1, train_fraction=tf, standardize=True, seed=None)

    n_v = X1_train.shape[-1] #Number of nodes in the visible layer
    n_h = 8 #Number of nodes in the hidden layer
    epochs = 100

    # Sample x values from P(x;h,params) and build theorical correlation matrix
    # Compute also the empirical correlation matrix on the train set for comparison
    variances = np.var(X1_train, axis=0, keepdims=True)
    grbm = learn_rbm(X1_train, X1_test, n_hid=n_h, epochs=epochs)
    sampling_steps = 400


    gibbsampler = GibbsSampler(grbm)
    ndreams = 1000
    X_dreams = dreams(ndreams, gibbsampler, n_v, k=sampling_steps)
    E_dreams = np.corrcoef(X_dreams.T)
    # E_dreams = ndreams**-1 * X_dreams.T @ X_dreams
    T = X1_train.shape[0]
    E_empir = np.corrcoef(X1_train.T)

    W = grbm.get_parameters()[0]

    J = W @ W.T
    # E_empir = T**-1 * X1_train.T @ X1_train
    # Saving results outside repository
    dream_dir = op.join(results_dir,'dream')
    np.save(op.join(dream_dir, 'E_dreams.npy'), E_dreams)
    np.save(op.join(dream_dir, 'E_empir.npy'), E_empir)
    np.save(op.join(dream_dir, 'J.npy'), J)
