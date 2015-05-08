#!/usr/bin/env python

"""This is implementaion of dimensionality reduction algorithms
"""

from norm import standardize
from parseData import textAsFloat
import numpy as np

def selectFeat(X, dim, judge):
    """ Select features using exhaustive search based on judgedment function

    Parameters
    ----------
    X:   
    """
    return None

def pca(dataset, k):
    """This is implementation of PCA

    parameters
    ----------
    dataset: the given data (not standardized)
    k:       the number of principle components

    returns
    -------
    kpcDataset: the 'standardized' k-principle-feature dataset
    """
    dataset = standardize(dataset, True, False)
    covMat = np.cov(dataset, rowvar=0)
    eigVals, eigVects = np.linalg.eig(covMat)
    eigIndex = np.argsort(-eigVals)[:k]
    kEigVects = eigVects[eigIndex]
    kpcDataset = dataset * kEigVects.T
    return kpcDataset

def main():
    X = textAsFloat('../testData/testPCA.txt', None, '\t')
    print pca(X, 1)

if __name__ == '__main__':
    main()
