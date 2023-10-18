import numpy
import matplotlib.pyplot as plt

def confusionMatrix(predictions, LTE):
    nClasses = len(set(LTE))
    confusionMatrix = numpy.zeros((nClasses,nClasses))
    for i in range(len(LTE)):
        confusionMatrix[predictions[i]][LTE[i]] += 1
    return confusionMatrix

def optimalBayesDecision(llr, pi, Cfn, Cfp, treshold=0):
    if treshold == 0:
        t = - numpy.log((pi*Cfn)/((1-pi)*Cfp))
    else:
        t = treshold
    return [1 if x > t else 0 for x in llr]

def bayesianRisk(confusionMatrix, pi, Cfn, Cfp):
    FNR = confusionMatrix[0][1]/(confusionMatrix[0][1] + confusionMatrix[1][1]) 
    FPR = confusionMatrix[1][0]/(confusionMatrix[0][0] + confusionMatrix[1][0])
    return pi*Cfn*FNR + (1 - pi)*Cfp*FPR

def normalizedDCF(confusionMatrix, pi, Cfn, Cfp):
    risk = bayesianRisk(confusionMatrix, pi, Cfn, Cfp)
    minDummy = numpy.min([pi*Cfn, (1-pi)*Cfp])
    return risk/minDummy

def minDCF(llrs, labels, pi, Cfn, Cfp):
    risks = []
    thresholds = numpy.sort(llrs)
    for i, llr in enumerate(thresholds):
        predictions = optimalBayesDecision(llrs, pi, Cfn, Cfp, llr)
        cMatrix = confusionMatrix(predictions, labels)
        risk = normalizedDCF(cMatrix, pi, Cfn, Cfp)
        risks.append(risk)
    return numpy.min(risks)

def ROC_plot(llrs, labels, pi, Cfn, Cfp):
    FNR=[]
    FPR=[]
    tresholds = numpy.sort(llrs)
    for t in tresholds:
        #I consider t (and not prior) as threshold value for split up the dataset in the two classes' set
        pred = optimalBayesDecision(llrs, pi, Cfn, Cfp, t)
        #I compute the confusion_matrix starting from these new results
        cm = confusionMatrix(pred, labels)
            
        FNR.append(cm[0,1]/(cm[0,1]+cm[1,1]))
        FPR.append(cm[1,0]/(cm[1,0]+cm[0,0]))
        
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    TPR=1-numpy.array(FNR)
    plt.plot(FPR,TPR,scalex=False,scaley=False)
    plt.show()