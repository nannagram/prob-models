import numpy as np
from numpy.random import default_rng
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


def train_rbm(X_train, X_test, n_hid, epochs, n_epochs=10):
    n_v = X_train.shape[-1]
    variances = np.var(X_train, axis=0, keepdims=True)
    
    ll_results, params = {}, {}
    ll_trains = np.zeros((len(n_hid), int(epochs/n_epochs)+1))
    ll_tests = np.zeros((len(n_hid), int(epochs/n_epochs)+1))
    count = 0 
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
            
            trainer.train(data = X_train)
            
            trainer.train(data=X_train)
            
            if epoch % 10 == 0:
                count_epoch += 1    
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

if __name__ == '__main__':
    # Load data 
    paths = read_paths(op.join(os.getcwd(), op.join('..', '..', 'paths.txt')))
    results_dir = op.join(os.getcwd(), 'results')
    # paths = read_paths('/home/jerry/python_projects/other/prob-models/paths.txt')
    datadic = getdata(paths, dataset='CamCAN')
    T, N = np.shape(list(datadic.values())[0])

    tf = 0.8
    # X1 = datadic[1]
    # X_train, X_test = [], []
    # for x in datadic.values():
    #     _X_train, _X_test = split_train_test(x, train_fraction=tf, 
    #                                          standardize=True, seed=None)
    #     X_train.append(_X_train)
    #     X_test.append(_X_test)
        
    # X_train = np.concatenate(X_train, axis=0)
    # X_test = np.concatenate(X_test, axis=0)
    for sj, x in datadic.items():
        results, params = single_rbm(x, n_hid=[25], tf=0.8, epochs=100)
        
        sub_results_dir = op.join(results_dir, f'sub_{sj}')
        if not op.exists(sub_results_dir):
            os.makedirs(sub_results_dir)
        results.to_csv(op.join(sub_results_dir, 'results.csv'))
        np.save(op.join(sub_results_dir, 'params.npy'), params)
