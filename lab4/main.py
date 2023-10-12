import numpy
import sys
sys.path.insert(0, 'Lab_ML/libs')
from utils import computeMean, computeCovarianceMatrix, vrow
from multivariateGaussianModel import loglilelihood, logpdf_GAU_ND
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    m = numpy.ones((1,1)) * 1.0
    C = numpy.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    #plt.show()

    XND = numpy.load("Lab_ML/lab4/XND.npy")
    muML = computeMean(XND)
    sigmaML = computeCovarianceMatrix(XND)
    ll = loglilelihood(XND, muML, sigmaML)
    print(ll)

    X1D = numpy.load("Lab_ML/lab4/X1D.npy")
    muML = computeMean(X1D)
    sigmaML = computeCovarianceMatrix(X1D)
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), muML, sigmaML)))
    plt.show()
    ll = loglilelihood(X1D, muML, sigmaML)
    print(ll)