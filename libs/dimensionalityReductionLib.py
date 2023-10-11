import scipy.linalg
import numpy as np
from utils import computeMean, computeCovarianceMatrix

def betweenClassCovarianceMatrix(D, L):
    N = D.shape[1]
    mu = computeMean(D)
    k = len(set(L))
    Sb = np.zeros((int(D.shape[0]), int(D.shape[0])))
    for c in range(k):
        Dc = D[:, L == c]
        nc = Dc.shape[1]
        muc = computeMean(Dc)
        Sb = Sb + nc * np.dot((muc - mu), (muc - mu).T)
    Sb = Sb / N
    return Sb

def withinClassCovarianceMatrix(D, L):
    N = D.shape[1]
    k = len(set(L))
    Sw = np.zeros((int(D.shape[0]), int(D.shape[0])))
    for c in range(k):
        Dc = D[:, L == c]
        nc = Dc.shape[1]
        Swc = computeCovarianceMatrix(Dc)
        Sw = Sw + nc*Swc
    Sw = Sw / N
    return Sw

def pcaProjection(D, m):
    C = computeCovarianceMatrix(D)
    U, s, Vh = np.linalg.svd(C)
    P = U[:, 0:m]
    DP = np.dot(P.T, D)
    return DP

def ldaProjection(D,L,m):
    SB = betweenClassCovarianceMatrix(D,L)
    SW = withinClassCovarianceMatrix(D,L)
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    DP = np.dot(W.T, D)
    return DP