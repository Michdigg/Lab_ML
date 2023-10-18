import scipy.optimize
import numpy
from utils import vcol

def f(x):
    y = x[0]
    z = x[1]
    f = pow(y+3, 2) + numpy.sin(y) + pow(z+1,2)
    return f

def fWGrad(x):
    y = x[0]
    z = x[1]
    dey = 2*(y+3) + numpy.cos(y)
    dez = 2*(z+1)
    return f(x), numpy.array([dey, dez])

def logreg_obj_wrap(DTR, LTR, l):
    def logreg_obj(v):
        loss = 0
        
        
        w,b = v[0:-1],v[-1]
        w = vcol(w)
        n = DTR.shape[1]
        regularization = (l / 2) * numpy.sum(w ** 2)
        for i in range(n):
            
            if (LTR[i:i+1] == 1):
                zi = 1
            else:
                zi=-1
            loss += numpy.logaddexp(0,-zi * (numpy.dot(w.T,DTR[:,i:i+1]) + b))
        
        J = regularization + (1 / n) * loss
        
        return J

    return logreg_obj

def binary_logr_modelTrained(DTR, LTR, l):
    x, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg_obj_wrap(DTR, LTR, l), numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
    b = x[-1]
    w = numpy.array(x[0:-1])
    return w,b

def binary_logrPredictions(w, b, DTE):
    S = []

    for i in range(DTE.shape[1]):
        x = DTE[:,i:i+1]
        x = numpy.array(x)
        S.append(numpy.dot(w.T,x) + b)
    
    S = [1 if x > 0 else 0 for x in S]
    return S