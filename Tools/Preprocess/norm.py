#!/usr/bin/env python

import numpy as np

def standardize(dataset, removeMean=True, scale=True):
    """Standardize the dataset to the Gassian Distribution
       with zero mean and unit variance

    parameters
    ----------
    dataset:    the given data, perform well if its distribution looks like Gaussian
    removeMean: if True (default), remove the mean for each individual feature
    scale:      if True (default), scale each feature by dividing its standard deviation

    returns
    -------
    dataset:   the standardized given dataset
    """
    dataset = np.mat(dataset)
    if removeMean:
        meanVec = dataset.mean(axis=0)
        dataset -= meanVec
    if scale:
        stdVec = dataset.std(axis=0)
        if 0 not in stdVec:
            dataset /= stdVec
        else:
            stdVec[stdVec==0] = 1
            dataset /= stdVec
    return dataset

def main():
    X = [[1., 3., 2.],
         [1., 4., -1.],
         [1., 5., 0.]]
    print standardize(X)

if __name__ == '__main__':
    main()
