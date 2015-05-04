#!/usr/bin/env python

import numpy as np
import Classifier.kNN as knn
import Classifier.errorEstimator as err
from itertools import combinations

def main():
    (Xtrain, ytrain, IDtrain, names) = readData('Training_Data.txt')
    # select3nnResubExaus(Xtrain, ytrain)
    # select3nnLooExaus(Xtrain, ytrain)
    select3nnResubForward(Xtrain, ytrain)
    select3nnLooForward(Xtrain, ytrain)
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

def select3nnResubExaus(Xtrain, ytrain):
    """ Select 1-3 feature using exhaustive search based on resubstitutions of 3nn
    """
    feats = range(Xtrain.shape[1])
    featNum = [1, 2, 3]
    for r in featNum:
        minErr = 1.0;
        for subsets in combinations(feats, r):
            subX = exaustiveSearch(Xtrain, subsets)
            model = knn.train(subX, ytrain, 3)
            yhat = knn.classify(subX, model)
            error = err.resub(ytrain, yhat)
            if error < minErr:
                bestSubsets = np.asarray(subsets) + 1
                minErr = error
        print 'the best susbsets with %d features is %s, with error %f' % (r, bestSubsets, minErr)
    return None

def select3nnLooExaus(Xtrain, ytrain):
    """ Select 1-3 feature using exhaustive search based on leave-one-out error of 3nn
    """
    feats = range(Xtrain.shape[1])
    featNum = [1, 2, 3]
    for r in featNum:
        minErr = 1.0;
        for subsets in combinations(feats, r):
            subX = exaustiveSearch(Xtrain, subsets)
            n = len(ytrain)
            yhat = np.empty(n, dtype=int)
            for i in range(n):
                Xtest = subX[i]
                modelX = np.delete(subX, i, 0)
                modelY = np.delete(ytrain, i, 0)
                yhat[i] = int(knn.kNN(modelX, modelY, Xtest, 3))
            error = err.resub(ytrain, yhat)
            if error < minErr:
                bestSubsets = np.asarray(subsets) + 1
                minErr = error
        print 'the best susbsets with %d features is %s, with error %f' % (r, bestSubsets.tolist(), minErr)
    return None

def select3nnResubForward(Xtrain, ytrain, maxFeatNum=5):
    """ Select 1-5 feature using forward search based on resubstitutions of 3nn
    """
    D = Xtrain.shape[1]
    featSets = []
    for i in range(maxFeatNum):
        minErr = 1.0
        for j in range(D):
            subsets = list(featSets)
            subsets.append(j)
            subX = exaustiveSearch(Xtrain, subsets)
            model = knn.train(subX, ytrain, 3)
            yhat = knn.classify(subX, model)
            error = err.resub(ytrain, yhat)
            if error < minErr:
                best = j+1
                minErr = error
        featSets.append(best)
        print 'the best susbsets with %d features is %s, with error %f' % (i+1, featSets, minErr)
    return None

def select3nnLooForward(Xtrain, ytrain, maxFeatNum=5):
    """ Select 1-5 feature using forward search based on leave-one-out error of 3nn
    """
    D = Xtrain.shape[1]
    featSets = []
    for i in range(maxFeatNum):
        minErr = 1.0
        for j in range(D):
            subsets = list(featSets)
            if (j+1) in featSets:
                continue;
            subsets.append(j)
            subX = exaustiveSearch(Xtrain, subsets)
            n = len(ytrain)
            yhat = np.empty(n, dtype=int)
            for k in range(n):
                Xtest = subX[k]
                modelX = np.delete(subX, k, 0)
                modelY = np.delete(ytrain, k, 0)
                yhat[k] = int(knn.kNN(modelX, modelY, Xtest, 3))
            error = err.resub(ytrain, yhat)
            if error < minErr:
                best = j+1
                minErr = error
        featSets.append(best)
        print 'the best susbsets with %d features is %s, with error %f' % (i+1, featSets, minErr)
    return None

def exaustiveSearch(X, subsets):
    N = X.shape[0]
    D = len(subsets)
    subX = np.zeros((N,D))
    for i in range(D):
        subX[:,i] = X[:, subsets[i]]
    return subX

if __name__ == '__main__':
    main()
