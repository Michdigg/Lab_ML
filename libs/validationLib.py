import numpy

def paramsEstimationAndPrediction(DTR, LTR, DTE, LTE, paramsEstimator, predictor):
    means, covariances = paramsEstimator(DTR, LTR)

    if (len(numpy.array(covariances).shape) == len(means)):
        predictedLabels = predictor(DTE, means, covariances)
    else:
        predictedLabels = predictor(DTE, means, covariances)

    predictions = len(predictedLabels)
    mispredictions = numpy.count_nonzero(predictedLabels - LTE)
    correctPredictions = predictions - mispredictions

    accuracy = (correctPredictions / predictions) * 100
    errRate = (mispredictions / predictions) * 100

    return accuracy, errRate

def kfoldCrossValidationGaussianModels(D, L, k, paramsEstimator, predictor):
    accuracies = []
    errRates = []
    nElements = int(numpy.ceil(D.shape[1] / k))

    for i in range(k):
        DTE = D[:, i:nElements*(i+1)]
        LTE = L[i:nElements*(i+1)]
        DTR = numpy.concatenate((D[:,0:i], D[:, nElements*(i+1):]), axis=1)
        LTR = numpy.concatenate((L[0:i], L[nElements*(i+1):]))

        accuracy, errRate = paramsEstimationAndPrediction(DTR, LTR, DTE, LTE, paramsEstimator, predictor)

        accuracies.append(accuracy)
        errRates.append(errRate)

    accuracy = numpy.array(accuracies).mean()
    errRate = numpy.array(errRates).mean()

    return accuracy, errRate


def leaveOneOutGaussianModels(D, L, paramsEstimator, predictor):
    accuracies = []
    errRates = []

    for i in range(D.shape[1]):
        DTE = D[:,i:i+1]
        LTE = L[i:i+1]
        DTR = numpy.delete(D,i,axis=1)
        LTR = numpy.delete(L,i)
        

        accuracy, errRate = paramsEstimationAndPrediction(DTR, LTR, DTE, LTE, paramsEstimator, predictor)

        accuracies.append(accuracy)
        errRates.append(errRate)

    accuracy = numpy.array(accuracies).mean()
    errRate = numpy.array(errRates).mean()

    return accuracy, errRate