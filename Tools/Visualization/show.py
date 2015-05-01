#!/usr/bin/env python

import parseData as pd
import numpy as np
import matplotlib.pyplot as plt
from Preprocess.DimReduct import pca

def showData(X, dim=2):
    plt.figure(1)
    plotData(X, dim)
    plt.show()
    return None

def plotData(X, dim=2):
    X = np.asarray(X)
    D = X.shape[1]
    if D>dim:
        print 'Input data is less then %d dimension:' % dim
    elif D<dim:
        X = pca(X,dim)
    if dim==2:
        plot2D(X)
    return None

def plot2D(X, args='ob'):
    plt.plot(X[:,0], X[:,1], args)
    return None

def main():
    X = pd.textAsFloat('../testData/testKmeans.txt', None, '\t')
    showData(X)

if __name__ == '__main__':
    main()
