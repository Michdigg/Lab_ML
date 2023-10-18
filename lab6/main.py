import sys
import numpy
sys.path.insert(0, 'Lab_ML/libs')
sys.path.insert(1, 'Lab_ML/lab6/data')
from data.load import load_data, split_data
from multinomailModels import createDictionary, multinomialPredictor, trainModel, normalize

if __name__ == "__main__":
    lInf, lPur, lPar = load_data()
    trainInf, testInf = split_data(lInf, 4)
    trainPur, testPur = split_data(lPur, 4)
    trainPar, testPar = split_data(lPar, 4)

    trainingSet = trainInf + trainPar + trainPur

    wordDict = createDictionary(trainingSet)

    infDict = trainModel(wordDict, trainInf)
    parDict = trainModel(wordDict, trainPar)
    purDict = trainModel(wordDict, trainPur)

    normalize(infDict, trainInf)
    normalize(parDict, trainPar)
    normalize(purDict, trainPur)

    llsInfInf = multinomialPredictor(infDict, testInf)
    llsPurInf = multinomialPredictor(parDict, testInf)
    llsParInf = multinomialPredictor(purDict, testInf)

    scoresInf = [llsInfInf, llsPurInf, llsParInf]
    scoresInf = numpy.array(scoresInf) * 1 / 3
    
    predictionsInf = numpy.argmax(scoresInf, axis=0)

    print('ciao')