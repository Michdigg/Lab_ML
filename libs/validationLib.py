import numpy

def leaveOneOutGaussianModels(D, L, paramsEstimator, predictor):
    accuracies = []
    errRates = []

    for i in range(D.shape[1]):
        DTE = D[:,i:i+1]
        LTE = L[i:i+1]
        DTR = numpy.delete(D,i,axis=1)
        LTR = numpy.delete(L,i)
        

        means, covariances = paramsEstimator(DTR, LTR)

        if (len(numpy.array(covariances).shape) == len(means)):
            predictedLabels = predictor(DTE, means, covariances)
        else:
            predictedLabels = predictor(DTE, means, covariances)

        predictions = len(predictedLabels)
        mispredictions = numpy.count_nonzero(predictedLabels - LTE)
        correctPredictions = predictions - mispredictions

        accuracies.append((correctPredictions / predictions) * 100)
        errRates.append((mispredictions / predictions) * 100)

    accuracy = numpy.array(accuracies).mean()
    errRate = numpy.array(errRates).mean()

    return accuracy, errRate