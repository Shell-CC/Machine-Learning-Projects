#!/usr/bin/env python

import numpy as np
import Classifier.kNN as knn
import Classifier.errorEstimator as err
import Classifier.gda as gda
from itertools import combinations
import matplotlib.pyplot as plt

def main():
    (Xtrain, ytrain, IDtrain, names) = readData('Training_Data.txt')
    # selectResubExaus(Xtrain, ytrain, 'knn')
    # selectLooExaus(Xtrain, ytrain, 'knn')
    # selectResubForward(Xtrain, ytrain, 'knn')
    # selectLooForward(Xtrain, ytrain, 'knn')

    # selectResubExaus(Xtrain, ytrain, 'dlda')
    selectLooExaus(Xtrain, ytrain, 'dlda')
    # selectResubForward(Xtrain, ytrain, 'dlda')
    # selectLooForward(Xtrain, ytrain, 'dlda')

    # subX = exaustiveSearch(Xtrain, (11,19))
    # plt.figure(1)
    # plot(subX, ytrain)
    # model = gda.ldatrain(subX, ytrain)
    # yhat = gda.ldaclassify(model, subX)
    # print err.resub(ytrain, yhat)
    # plt.figure(2)
    # plot(subX, yhat)
    # plt.show()
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

def selectResubExaus(Xtrain, ytrain, method):
    """ Select 1-3 feature using exhaustive search based on resubstitutions
        of knn/DLDA/linear SVM
    """
    feats = range(Xtrain.shape[1])
    featNum = [1, 2, 3]
    for r in featNum:
        minErr = 1.0;
        for subsets in combinations(feats, r):
            subX = exaustiveSearch(Xtrain, subsets)
            if method=='knn':
                model = knn.train(subX, ytrain, 3)
                yhat = knn.classify(subX, model)
            elif method=='dlda':
                model = gda.ldatrain(subX, ytrain)
                yhat = gda.ldaclassify(model, subX)
            error = err.resub(ytrain, yhat)
            if error < minErr:
                bestSubsets = np.asarray(subsets) + 1
                minErr = error
        print 'the best susbsets with %d features is %s, with error %f' % (r, bestSubsets, minErr)
    return None

def selectLooExaus(Xtrain, ytrain, method):
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
                if method=='knn':
                    model = knn.train(modelX, modelY, 3)
                    yhat[i] = knn.classify(Xtest, model)
                elif method=='dlda':
                    model = gda.ldatrain(modelX, modelY)
                    yhat[i] = gda.ldaclassify(model, Xtest)
            error = err.resub(ytrain, yhat)
            if error < minErr:
                bestSubsets = np.asarray(subsets) + 1
                minErr = error
        print 'the best susbsets with %d features is %s, with error %f' % (r, bestSubsets.tolist(), minErr)
    return None

def selectResubForward(Xtrain, ytrain, method, maxFeatNum=5):
    """ Select 1-5 feature using forward search based on resubstitutions
        of 3NN/DLDA/
    """
    D = Xtrain.shape[1]
    featSets = []
    for i in range(maxFeatNum):
        minErr = 1.0
        for j in range(D):
            subsets = list(featSets)
            if j in featSets:
                continue;
            subsets.append(j)
            subX = exaustiveSearch(Xtrain, subsets)
            if method=='knn':
                model = knn.train(subX, ytrain, 3)
                yhat = knn.classify(subX, model)
            elif method=='dlda':
                model = gda.ldatrain(subX, ytrain)
                yhat = gda.ldaclassify(model, subX)
            error = err.resub(ytrain, yhat)
            if error < minErr:
                bestSubsets = np.asarray(subsets)+1
                minErr = error
        featSets = (bestSubsets-1).tolist()
        print 'the best susbsets with %d features is %s, with error %f' % (i+1, bestSubsets.tolist(), minErr)
    return None

def selectLooForward(Xtrain, ytrain, method, maxFeatNum=5):
    """ Select 1-5 feature using forward search based on leave-one-out error
    """
    D = Xtrain.shape[1]
    featSets = []
    for i in range(maxFeatNum):
        minErr = 1.0
        for j in range(D):
            subsets = list(featSets)
            if j in featSets:
                continue;
            subsets.append(j)
            subX = exaustiveSearch(Xtrain, subsets)
            n = len(ytrain)
            yhat = np.empty(n, dtype=int)
            for k in range(n):
                Xtest = subX[k]
                modelX = np.delete(subX, k, 0)
                modelY = np.delete(ytrain, k, 0)
                if method=='knn':
                    # yhat[k] = int(knn.kNN(modelX, modelY, Xtest, 3))
                    model = knn.train(modelX, modelY, 3)
                    yhat[k] = knn.classify(Xtest, model)
                elif method=='dlda':
                    model = gda.ldatrain(modelX, modelY)
                    yhat[k] = gda.ldaclassify(model, Xtest)
            error = err.resub(ytrain, yhat)
            if error < minErr:
                bestSubsets = np.asarray(subsets)+1
                minErr = error
        featSets = (bestSubsets-1).tolist()
        print 'the best susbsets with %d features is %s, with error %f' % (i+1, bestSubsets.tolist(), minErr)
    return None

def exaustiveSearch(X, subsets):
    N = X.shape[0]
    D = len(subsets)
    subX = np.zeros((N,D))
    for i in range(D):
        subX[:,i] = X[:, subsets[i]]
    return subX

def plot(X, labels, K=2):
    colors = ['ob', 'or', 'og', 'oc', 'om', '^b', '^r', '^g', '^c', '^m']
    clusters = [[] for _ in xrange(K)]
    N = len(X)
    for i in range(N):
        clusters[labels[i]].append(X[i].tolist())
    for k in range(K):
        c = np.asarray(clusters[k])
        plt.plot(c[:,0], c[:,1], colors[k])
    return None

if __name__ == '__main__':
    main()
