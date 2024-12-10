import numpy as np
from numpy.random import default_rng
import random
from pydeep.rbm.model import GaussianBinaryRBM
import os
import os.path as op
import pandas as pd
from prob_models.utils.utils import read_paths, getdata
from pydeep.rbm.trainer import CD
from pydeep.rbm.estimator import (partition_function_factorize_h, 
                                  log_likelihood_v, 
                                  reconstruction_error,
                                  annealed_importance_sampling)
from pydeep.rbm.sampler import GibbsSampler

def split_train_test(X, train_fraction, standardize=False, shuffle=True, 
                     seed=None):
    rng = default_rng(seed)
    X = X.copy()
    if shuffle:
        rng.shuffle(X)
    T, N = X.shape
    Xtrain = X[:int(train_fraction*T)]
    Xtest = X[int(train_fraction*T):]
    if standardize:
        avg = np.mean(Xtrain, axis=0)
        Xtrain -= avg 
        std = np.std(Xtrain, axis=0)
        Xtrain /= std
        
        Xtest -= avg
        Xtest /= std


    return Xtrain, Xtest

# Train model and return the model class
# Find a better name!
def learn_rbm(X_train, X_test, n_hid, epochs):
    n_v = X_train.shape[-1]
    variances = np.var(X_train, axis=0, keepdims=True)
    
                
    grbm = GaussianBinaryRBM(number_visibles=n_v,
                            number_hiddens=n_hid,
                            data=X_train,
                            initial_sigma=variances,
                            initial_visible_offsets=0.0,
                            initial_hidden_offsets=0.0)
    trainer = CD(grbm)
    
    for epoch in range(epochs):
        
        trainer.train(data=X_train, epsilon=0.1)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}')
    return grbm
def train_rbm(X_train, X_test, n_hid, epochs):
    n_v = X_train.shape[-1]
    variances = np.var(X_train, axis=0, keepdims=True)
    
    ll_results, params = {}, {}
    for n_h in n_hid:
        print(f'\nTraining with {n_h} hidden units')
                
        grbm = GaussianBinaryRBM(number_visibles=n_v,
                                number_hiddens=n_h,
                                data=X_train,
                                initial_sigma=variances,
                                initial_visible_offsets=0.0,
                                initial_hidden_offsets=0.0)
        trainer = CD(grbm)
        
        for epoch in range(epochs):
            
            trainer.train(data=X_train)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}')
        
        print('Computing log-likelihood and reconstruction error...')
        # log_z = partition_function_factorize_h(grbm, status=True)
        log_z = annealed_importance_sampling(grbm, status=False)
        ll_train = np.mean(log_likelihood_v(grbm, log_z, X_train))
        ll_test = np.mean(log_likelihood_v(grbm, log_z, X_test))
        re = np.mean(reconstruction_error(grbm, X_train))
        print('...done')
        
        ll_results[n_h] = {'ll_train': ll_train, 'll_test': ll_test, 're': re}
        params[n_h] = grbm.get_parameters()[0]
        
    ll_results = pd.DataFrame.from_dict(ll_results, orient='index')
    return ll_results, params

def single_rbm(X, n_hid, tf=0.8, epochs=100):
    X_train, X_test = split_train_test(X, train_fraction=tf, standardize=True, seed=None)
    results, paramas = train_rbm(X_train, X_test, n_hid, epochs)
    return results, paramas

# Samples the visible variables from conditional distribution.
def dreams(v, ndreams,gibbssampler, n_v,k=400):
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
    # paths = read_paths('/home/jerry/python_projects/other/prob-models/paths.txt')
    datadic = getdata(paths, dataset='CamCAN')
    T, N = np.shape(list(datadic.values())[0])

    tf = 0.8

    ##########################################################################
    # Sample x values from P(x;h,params) and build theorical correlation matrix
    # Compute also the empirical correlation matrix on the train set for comparison
    X1 = datadic[1]
    X1_train, X1_test = split_train_test(X1, train_fraction=tf, standardize=True, seed=None)
    n_v = X1_train.shape[-1]
    n_h = 8
    variances = np.var(X1_train, axis=0, keepdims=True)
    grbm = learn_rbm(X1_train, X1_test, n_hid=n_h, epochs=100)


    gibbsampler = GibbsSampler(grbm)
    ndreams = 1000
    idx = np.random.choice(X1_train.shape[0])
    v_start = X1_train[1,:]
    X_dreams = dreams(v_start, ndreams, gibbsampler, n_v, k=400)
    E_dreams = np.corrcoef(X_dreams.T)
    # E_dreams = ndreams**-1 * X_dreams.T @ X_dreams
    T = X1_train.shape[0]
    E_empir = np.corrcoef(X1_train.T)

    W = grbm.get_parameters()[0]

    J = W @ W.T
    # E_empir = T**-1 * X1_train.T @ X1_train
    # Saving results outside repository
    save_path = op.dirname(op.dirname(op.dirname(os.getcwd())))
    print(save_path)
    np.save(op.join(save_path, 'E_dreams.npy'), E_dreams)
    np.save(op.join(save_path, 'E_empir.npy'), E_empir)
    np.save(op.join(save_path, 'J.npy'), J)
###################################################################################
    '''Workflow for all subjects'''

    # for x in datadic.values():
    #     X_train, X_test = [], []
    #     for x in datadic.values():
    #         _X_train, _X_test = split_train_test(x, train_fraction=tf, 
    #                                             standardize=True, seed=None)
    #         X_train.append(_X_train)
    #         X_test.append(_X_test)
            
    #     X_train = np.concatenate(X_train, axis=0)
    #     X_test = np.concatenate(X_test, axis=0)

##################################################################################
    ''' Workflow for single subject '''

    # for sj, x in datadic.items():
    #     results, params = single_rbm(x, n_hid=[25], tf=0.8, epochs=100)
        
    #     sub_results_dir = op.join(results_dir, f'sub_{sj}')
    #     if not op.exists(sub_results_dir):
    #         os.makedirs(sub_results_dir)
    #     results.to_csv(op.join(sub_results_dir, 'results.csv'))
    #     np.save(op.join(sub_results_dir, 'params.npy'), params)
