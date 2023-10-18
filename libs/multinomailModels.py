import numpy

def createDictionary(trainl):
    wordCount = {}
    for i in range(len(trainl)):
        for j, word in enumerate(trainl[i].split()):
            if wordCount.get(word) == None:
                wordCount[word] = 0
    return wordCount

def trainModel(dictionary, trainl):
    wordCount = dictionary.copy()
    for i, line in enumerate(trainl):
        for j, word in enumerate(line.split()):
            wordCount[word] = wordCount.get(word) + 1
    return wordCount

def normalize(wordDict, trainl):
    nWords = 0
    for i, line in enumerate(trainl):
        for j, word in enumerate(line.split()):
            nWords = nWords + 1
    for i, word in enumerate(wordDict):
        wordDict[word] = wordDict.get(word) / nWords
    return wordDict
    
def multinomialPredictor(d, testl):
    logPosteriorProbabilities = []
    for i in range(len(testl)):
        logPosteriorProbability = []
        for j, word in enumerate(testl[i].split(" ")):
            if d.get(word) == None:
                logPosteriorProbability.append(numpy.log(0.001))
            else:
                logPosteriorProbability.append(numpy.log(d[word] + 0.001))
        logPosteriorProbabilities.append(numpy.array(logPosteriorProbability).sum())

    return logPosteriorProbabilities