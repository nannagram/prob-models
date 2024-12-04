import numpy as np
import pydeep
from pydeep.rbm.model import GaussianBinaryVarianceRBM
from joblib import Parallel, delayed

def n_hid_parallel(data, n_hid=np.arange(2, 101), n_jobs=2):
        Parallel(n_jobs=n_jobs)(delayed(n_hid_single)(data, n_hid) for n_hid in n_hid)
    return