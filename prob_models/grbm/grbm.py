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


def split_train_test(X, train_fraction, standardize=False, shuffle=True, seed=None):
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


def train_rbm(X_train, X_test, n_hid, epochs):
    n_v = X_train.shape[-1]
    variances = np.var(X_train, axis=0, keepdims=True)
    
    results = {}
    for n_h in n_hid:
        print(f'\nTraining with {n_h} hidden units')
        
        grbm = GaussianBinaryRBM(number_visibles=n_v,
                                number_hiddens=n_h,
                                data=X1_train,initial_sigma=variances,
                                initial_visible_offsets=0.0,
                                initial_hidden_offsets=0.0)
        trainer = CD(grbm)
        
        for epoch in range(epochs):
            trainer.train(data = X_train)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}')
        
        print('Computing log-likelihood and reconstruction error...')
        # log_z = partition_function_factorize_h(grbm, status=True)
        log_z = annealed_importance_sampling(grbm, status=True)
        ll_train = np.mean(log_likelihood_v(grbm, log_z, X_train))
        ll_test = np.mean(log_likelihood_v(grbm, log_z, X_test))
        re = np.mean(reconstruction_error(grbm, X_train))
        print('...done')
        
        results[n_h] = {'ll_train': ll_train, 'll_test': ll_test, 're': re}
        
    results = pd.DataFrame.from_dict(results, orient='index')
    return results


if __name__ == '__main__':
    # Load data 
    paths = read_paths(op.join(os.getcwd(), op.join('paths.txt')))
    datadic = getdata(paths, dataset='CamCAN')
    T, N = np.shape(list(datadic.values())[0])

    tf = 0.8
    X1 = datadic[1]

    X1_train, X1_test = split_train_test(X1, train_fraction=tf, standardize=True, seed=0)
    print(f'X1 shape: {X1_train.shape}')

    n_hid = np.arange(2, 10, 1)
    epochs = 100
    results = train_rbm(X1_train, X1_test, n_hid, epochs)
    print(results)

# Load data 
paths = read_paths(op.join(os.getcwd(), op.join( 'paths.txt')))
#paths = read_paths(op.join(os.getcwd(), op.join('..','..', 'paths.txt')))

print ('loading data')
datadic = getdata(paths,dataset='CamCAN')

T, N = np.shape(list(datadic.values())[0])

tf = 0.8
sj_names = [name for name in datadic.keys()]
X1 = datadic[1]

X1_train, X1_test = split_train_test(X1, train_fraction=tf, standardize=True, seed=0)

n_v = X1_train.shape[-1]
print (np.shape(X1_train),np.shape(X1_test))

n_h = 20
variances = np.var(X1_train,axis=0,keepdims=True)
grbm = GaussianBinaryRBM(number_visibles=n_v,number_hiddens=n_h,
                         data=X1_train,initial_sigma=variances,
                         initial_visible_offsets=0.0,
                         initial_hidden_offsets=0.0)

trainer = CD(grbm)

epochs = 50
for epoch in range(epochs):
    trainer.train(data=X1_train)

log_z = partition_function_factorize_h(grbm)
ll_train = np.mean(log_likelihood_v(grbm, log_z, X1_train))
ll_test = np.mean(log_likelihood_v(grbm, log_z, X1_test))
re = np.mean(reconstruction_error(grbm, X1_train))

print(ll_train)
print(ll_test)
print(re)