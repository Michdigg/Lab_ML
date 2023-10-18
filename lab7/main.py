import sys
import numpy
sys.path.insert(0, 'Lab_ML/libs')
from binaryLogisticRegression import f, fWGrad, binary_logr_modelTrained, binary_logrPredictions
from logisticRegression import logr_modelTrained, logrPredictions
import scipy.optimize
import sklearn.datasets

def load_iris():
    D,L = sklearn.datasets.load_iris()['data'].T,sklearn.datasets.load_iris()['target']
    return D,L

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2) return D, L
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)

if __name__ == "__main__":
    x1, fmin1, d1 = scipy.optimize.fmin_l_bfgs_b(f, numpy.array([0,0]), approx_grad=True)
    x2, fmin2, d2 = scipy.optimize.fmin_l_bfgs_b(fWGrad, numpy.array([0,0]))

    D,L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    l = 0.000001

    w1, b1 = binary_logr_modelTrained(DTR, LTR, l)
    binaryPredictions = binary_logrPredictions(w1, b1, DTE)
    er1 = numpy.count_nonzero(binaryPredictions - LTE)/len(LTE) * 100

    D,L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    w2, b2 = logr_modelTrained(DTR, LTR, l)
    predictions = logrPredictions(w2, b2, DTE)
    er2 = numpy.count_nonzero(predictions - LTE)/len(LTE) * 100

    print(er1)
    print(er2)