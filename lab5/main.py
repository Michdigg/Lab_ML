import numpy
import sys
sys.path.insert(0, 'Lab_ML/libs')
from gaussianClassification import gaussianParamsEstimator, gaussianLabelPredict, optimizedGaussianLabelPredict, naiveBayesGaussianParamsEstimator, tiedGaussianParamsEstimator, optimizedTiedGaussianLabelPredict, tiedNaiveBayesGaussianParamsEstimator
from validationLib import leaveOneOutGaussianModels
import sklearn.datasets as sk

def load_iris():
    D,L = sk.load_iris()['data'].T,sk.load_iris()['target']
    return D,L

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
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    ##MVG model

    #parameters estimation with training data
    means, covariances = gaussianParamsEstimator(DTR, LTR)
    #prediction with test data
    predictedLabels = gaussianLabelPredict(DTE, means, covariances)

    predictions = len(predictedLabels)
    mispredictions = numpy.count_nonzero(predictedLabels - LTE)
    correctPredictions = predictions - mispredictions

    accuracy = (correctPredictions / predictions) * 100
    err = (mispredictions / predictions) * 100

    #prediction with test data optimized
    optPredictedLabels = optimizedGaussianLabelPredict(DTE, means, covariances)

    ## Naive Bayes
    #parameters estimation with training data
    nbMeans, nbCovariances = naiveBayesGaussianParamsEstimator(DTR, LTR)
    #prediction with test data
    nbPredictedLabels = optimizedGaussianLabelPredict(DTE, nbMeans, nbCovariances)

    nbPredictions = len(nbPredictedLabels)
    nbMispredictions = numpy.count_nonzero(nbPredictedLabels - LTE)
    nbCorrectPredictions = nbPredictions - nbMispredictions

    nbAccuracy = (nbCorrectPredictions / nbPredictions) * 100
    nbErr = (nbMispredictions / nbPredictions) * 100

    ## Tied
    #parameters estimation with training data
    tiedMeans, tiedCovariance = tiedGaussianParamsEstimator(DTR, LTR)
    #prediction with test data
    tiedPredictedLabels = optimizedTiedGaussianLabelPredict(DTE, tiedMeans, tiedCovariance)

    tiedPredictions = len(tiedPredictedLabels)
    tiedMispredictions = numpy.count_nonzero(tiedPredictedLabels - LTE)
    tiedCorrectPredictions = tiedPredictions - tiedMispredictions

    tiedAccuracy = (tiedCorrectPredictions / tiedPredictions) * 100
    tiedErr = (tiedMispredictions / tiedPredictions) * 100

    ## Tied Naive Bayes
    #parameters estimation with training data
    tiedNBMeans, tiedNBCovariance = tiedNaiveBayesGaussianParamsEstimator(DTR, LTR)
    #prediction with test data
    tiedNBPredictedLabels = optimizedTiedGaussianLabelPredict(DTE, tiedNBMeans, tiedNBCovariance)

    tiedNBPredictions = len(tiedNBPredictedLabels)
    tiedNBMispredictions = numpy.count_nonzero(tiedNBPredictedLabels - LTE)
    tiedNBCorrectPredictions = tiedNBPredictions - tiedNBMispredictions

    tiedNBAccuracy = (tiedNBCorrectPredictions / tiedNBPredictions) * 100
    tiedNBErr = (tiedNBMispredictions / tiedNBPredictions) * 100

    accuracy1, errRate1 = leaveOneOutGaussianModels(D, L, gaussianParamsEstimator, optimizedGaussianLabelPredict)
    accuracy2, errRate2 = leaveOneOutGaussianModels(D, L, naiveBayesGaussianParamsEstimator, optimizedGaussianLabelPredict)
    accuracy3, errRate3 = leaveOneOutGaussianModels(D, L, tiedGaussianParamsEstimator, optimizedTiedGaussianLabelPredict)
    accuracy4, errRate4 = leaveOneOutGaussianModels(D, L, tiedNaiveBayesGaussianParamsEstimator, optimizedTiedGaussianLabelPredict)

    print('End of file')