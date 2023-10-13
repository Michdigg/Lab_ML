import numpy
import scipy.special
from utils import computeMean, computeCovarianceMatrix, vrow
from multivariateGaussianModel import logpdf_GAU_ND
from dimensionalityReductionLib import withinClassCovarianceMatrix

def gaussianParamsEstimator(DT, LT):
    k = len(set(LT))
    meansVector = []
    covariancesVector = []
    for c in range(k):
        Dc = DT[:, LT == c]
        muc = computeMean(Dc)
        Cc = computeCovarianceMatrix(Dc)
        meansVector.append(muc)
        covariancesVector.append(Cc)
    return meansVector, covariancesVector

def naiveBayesGaussianParamsEstimator(DT, LT):
    k = len(set(LT))
    meansVector = []
    covariancesVector = []
    for c in range(k):
        Dc = DT[:, LT == c]
        muc = computeMean(Dc)
        Cc = numpy.diag(numpy.diag(computeCovarianceMatrix(Dc)))
        meansVector.append(muc)
        covariancesVector.append(Cc)
    return meansVector, covariancesVector

def tiedGaussianParamsEstimator(DT, LT):
    k = len(set(LT))
    meansVector = []
    for c in range(k):
        Dc = DT[:, LT == c]
        muc = computeMean(Dc)
        meansVector.append(muc)
    return meansVector, withinClassCovarianceMatrix(DT, LT)

def tiedNaiveBayesGaussianParamsEstimator(DT, LT):
    k = len(set(LT))
    meansVector = []
    for c in range(k):
        Dc = DT[:, LT == c]
        muc = computeMean(Dc)
        meansVector.append(muc)
    return meansVector, numpy.diag(numpy.diag(withinClassCovarianceMatrix(DT, LT)))

def gaussianScoreEvaluator(DE, means, covariances, Pc = []):
    S = []
    k = len(means)
    for c in range(k):
        pdf = logpdf_GAU_ND(DE, means[c], covariances[c])
        S.append(pdf)
    Sjoint = []
    if len(Pc) == 0:
        P = 1 / k
        Sjoint = numpy.exp(numpy.array(S)) * P
    else:
        #todo
        print('Prior probability case to implement')
    
    return Sjoint

def optGaussianScoreEvaluator(DE, means, covariances, Pc = []):
    S = []
    k = len(means)
    for c in range(k):
        pdf = logpdf_GAU_ND(DE, means[c], covariances[c])
        S.append(pdf)
    logSJoint = []
    if len(Pc) == 0:
        P = 1 / k
        logSJoint = numpy.array(S) + numpy.log(P)
    else:
        #todo
        print('Prior probability case to implement')
    
    return logSJoint

def optTiedGaussianScoreEvaluator(DE, means, tiedCovariance, Pc = []):
    S = []
    k = len(means)
    for c in range(k):
        pdf = logpdf_GAU_ND(DE, means[c], tiedCovariance)
        S.append(pdf)
    logSJoint = []
    if len(Pc) == 0:
        P = 1 / k
        logSJoint = numpy.array(S) + numpy.log(P)
    else:
        #todo
        print('Prior probability case to implement')
    
    return logSJoint

def gaussianLabelPredict(DE, means, covariances, Pc = []):
    SJoint = gaussianScoreEvaluator(DE, means, covariances)
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    return numpy.argmax(SPost, axis=0)

def optimizedGaussianLabelPredict(DE, means, covariances, Pc = []):
    logSJoint = optGaussianScoreEvaluator(DE, means, covariances)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)
    return numpy.argmax(SPost, axis=0)

def optimizedTiedGaussianLabelPredict(DE, means, tiedCovariance, Pc = []):
    logSJoint = optTiedGaussianScoreEvaluator(DE, means, tiedCovariance)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)
    return numpy.argmax(SPost, axis=0)