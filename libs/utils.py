import numpy as np

def load(fName='Lab_ML/iris.csv'):
    labelsDictionary = {
        'Iris-setosa':0,
        'Iris-versicolor':1,
        'Iris-virginica':2
    }

    attributesMatrix = []
    labels = []

    with open(fName,'r') as f:
        for i,row in enumerate(f):
            measures = row.split(',')
            labels.append(labelsDictionary[measures.pop()[:-1]])
            attributesMatrix = attributesMatrix + measures
    
    return np.array(attributesMatrix).astype(np.float32).reshape((150,4)).T, np.array(labels, dtype=np.int32)

def centerData(D):
    mu = mcol(D.mean(1))
    return D - mu

def computeMean(D):
    return mcol(D.mean(1))

def mcol(array):
    return array.reshape((array.shape[0], 1))

def mrow(array):
    return array.reshape((1, array.shape[0]))

def computeCovarianceMatrix(D):
    Dc = D - computeMean(D)
    return np.dot(Dc, Dc.T) / D.shape[1]