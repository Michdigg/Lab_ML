import scipy.optimize
import numpy as np
from utils import vcol

def logreg_obj_wrap(DTR, LTR, l):
    def logreg_obj(v):
        D = DTR.shape[0]
        n = DTR.shape[1]
        K = np.max(LTR)+1
         
        W = v[:D*K].reshape((D, K))
        b = v[D*K:]
        
        T = np.zeros((K,n))
        T[LTR, np.arange(n)] = 1

        first = (l / 2) * (W*W).sum()
        
        S = np.dot(W.T,DTR) + b[:, np.newaxis]
        
        lse = np.log(np.sum(np.exp(S), axis=0))
        Y_log = S - lse
        
        second = np.sum(T*Y_log)/n
        J = first - second
        return J

    return logreg_obj

def logr_modelTrained(DTR, LTR, l):
    nclasses = len(np.unique(LTR))
    x, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg_obj_wrap(DTR, LTR, l), np.zeros(DTR.shape[0] * nclasses + nclasses), approx_grad=True)
    b = x[DTR.shape[0]*nclasses:]
    w = np.array(x[:DTR.shape[0]*nclasses]).reshape((DTR.shape[0],nclasses))
    return w,b

def logrPredictions(w, b, DTE):
    S = np.dot(w.T, DTE) + b[:, np.newaxis]
    return np.argmax(S, axis=0)