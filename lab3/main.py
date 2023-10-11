import sys
sys.path.insert(0, 'Lab_ML/libs')
from utils import load
from dimensionalityReductionLib import pcaProjection, ldaProjection
import matplotlib.pyplot as plt

def plotProjection(projection):
    D0 = projection[:,labels==0]
    D1 = projection[:,labels==1]
    D2 = projection[:,labels==2]

    plt.figure()
    plt.xlabel('first principal direction')
    plt.ylabel('second principal direction')
    plt.scatter(D0[0,:],D0[1,:],label="Iris setosa")
    plt.scatter(D1[0,:],D1[1,:],label="Iris versicolor")
    plt.scatter(D2[0,:],D2[1,:],label="Iris verginica")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dataMatrix, labels = load()
    pcaProjection2d = pcaProjection(dataMatrix, 2)
    ldaProjection2d = ldaProjection(dataMatrix, labels, 2)
    
    plotProjection(pcaProjection2d)
    plotProjection(ldaProjection2d)


