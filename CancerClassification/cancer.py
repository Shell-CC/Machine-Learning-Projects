#!/usr/bin/env python

import numpy as np
import Classifier.kNN as knn
import Classifier.errorEstimator as err
from itertools import combinations

def main():
    (Xtrain, ytrain, IDtrain, names) = readData('Training_Data.txt')
    """ Select 1-3 feature using exhaustive search based on resubstitutions of 3nn
    """
    feats = range(len(names))
    featNum = [1, 2, 3]
    for r in featNum:
        minErr = 1.0;
        for subsets in combinations(feats, r):
            subX = exaustiveSearch(Xtrain, subsets)
            model = knn.train(subX, ytrain, 3)
            yhat = knn.classify(subX, model)
            # print ytrain
            # print yhat
            error = err.resub(ytrain, yhat)
            if error < minErr:
                bestSubsets = np.asarray(subsets) + 1
                minErr = error
        print 'the best susbsets with %d features is %s, with error %f' % (r, bestSubsets, minErr)
    # yhat = knn.test(subX, ytrain, model)
    (Xtest, ytest, IDtest, names) = readData('Testing_Data.txt')

def readData(filename):
    X = []; labels = []; IDs =[]
    f = open(filename, 'rU')
    names = f.readline().split()[1:-1]
    for line in f:
        data = line.split()
        IDs.append(int(data[0]))
        X.append(map(float, data[1:-1]))
        labels.append(int(data[-1]))
    f.close
    X = np.asarray(X)
    labels = np.asarray(labels)
    # print 'Read %d datas from %s.' % (len(IDs), filename)
    # print 'Each data contains %d genes, with a ID and a label.' % len(names)
    return X, labels, IDs, names

def exaustiveSearch(X, subsets):
    N = X.shape[0]
    D = len(subsets)
    subX = np.zeros((N,D))
    for i in range(D):
        subX[:,i] = X[:, subsets[i]]
    return subX

if __name__ == '__main__':
    main()
