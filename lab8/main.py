import sys
import numpy
sys.path.insert(0, 'Lab_ML/libs')
sys.path.insert(0, 'Lab_ML/lab8/Data')
import sklearn.datasets as sk
from gaussianClassification import gaussianParamsEstimator, optimizedGaussianLabelPredict, tiedGaussianParamsEstimator, optimizedTiedGaussianLabelPredict
from riskUtils import confusionMatrix, optimalBayesDecision, bayesianRisk, normalizedDCF, minDCF, ROC_plot


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

    means, covariances = gaussianParamsEstimator(DTR, LTR)
    predictedLabels = optimizedGaussianLabelPredict(DTE, means, covariances)
    confusionMatrixMVG = confusionMatrix(predictedLabels, LTE)

    tiedMeans, tiedCovariance = tiedGaussianParamsEstimator(DTR, LTR)
    tiedPredictedLabels = optimizedTiedGaussianLabelPredict(DTE, tiedMeans, tiedCovariance)
    confusionMatrixTied = confusionMatrix(tiedPredictedLabels, LTE)    

    commedia_ll = numpy.load("Lab_ML/lab8/Data/commedia_ll.npy")
    commedia_labels = numpy.load("Lab_ML/lab8/Data/commedia_labels.npy")
    commedia_pred = numpy.argmax(commedia_ll, axis=0)
    commedia_confusion_matrix = confusionMatrix(commedia_pred, commedia_labels)

    pi = 0.8
    Cfn = 1
    Cfp = 10
    llr_infpar = numpy.load("Lab_ML/lab8/Data/commedia_llr_infpar.npy")
    commedia_infpar_labels = numpy.load("Lab_ML/lab8/Data/commedia_labels_infpar.npy")
    infpar_predictions = optimalBayesDecision(llr_infpar, pi, Cfn, Cfp)
    infpar_confusion_matrix = confusionMatrix(infpar_predictions, commedia_infpar_labels)
    bayesianRiskInfPar = bayesianRisk(infpar_confusion_matrix, pi, Cfn, Cfp)
    normalizedBayesianRiskInfPar = normalizedDCF(infpar_confusion_matrix, pi, Cfn, Cfp)
    minDCFinfPar = minDCF(llr_infpar, commedia_infpar_labels, pi, Cfn, Cfp)
    ROC_plot(llr_infpar, commedia_infpar_labels, pi, Cfn, Cfp)

    print("")