import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'Lab_ML/libs')
from utils import load, centerData

def plot(D,L):
    D0 = D[:,L==0]
    D1 = D[:,L==1]
    D2 = D[:,L==2]

    featuresDictionary = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    for i in range(4):
        plt.figure()
        plt.xlabel(featuresDictionary[i])
        plt.ylabel("number of elements")
        plt.hist(D0[i,:],density=True,alpha = 0.7,label = "Iris-setosa")
        plt.hist(D1[i,:],density=True,alpha = 0.7,label = "Iris-versicolor")
        plt.hist(D2[i,:],density=True,alpha = 0.7,label = "Iris-verginica")
        plt.legend()
        plt.show()
        for j in range(4):
            if i == j:
                continue
            plt.figure()
            plt.xlabel(featuresDictionary[i])
            plt.ylabel(featuresDictionary[j])
            plt.scatter(D0[i,:],D0[j,:],label="Iris setosa")
            plt.scatter(D1[i,:],D1[j,:],label="Iris versicolor")
            plt.scatter(D2[i,:],D2[j,:],label="Iris verginica")
            plt.legend()
            plt.show()
    return

if __name__ == "__main__":
    dataMatrix, labels = load('Lab_ML/iris.csv')
    centeredDataMatrix = centerData(dataMatrix)
    plot(dataMatrix,labels)
    plot(centeredDataMatrix, labels)