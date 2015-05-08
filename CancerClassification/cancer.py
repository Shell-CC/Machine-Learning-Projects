#!/usr/bin/env python

import numpy as np
import Classifier.kNN as knn
import Classifier.errorEstimator as err
import Classifier.gda as gda
# import Classifier.svm as svm
from sklearn import svm
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    (Xtrain, ytrain, IDtrain, names) = readData('Training_Data.txt')
    (Xtest, ytest, IDtest, names) = readData('Testing_Data.txt')

    print 'Classification rule: 3NN'
    selectResubExaus(Xtrain, ytrain, 'knn')
    selectResubForward(Xtrain, ytrain, 'knn')
    selectLooExaus(Xtrain, ytrain, 'knn')
    selectLooForward(Xtrain, ytrain, 'knn')
    knnSets = [(12,),(7,66),(1,8,49),(12,),(12,22),(5,12,22),(5,12,22,24),(5,12,22,24,35), (49,), (7,66), (2,37,64), (49,), (37, 49), (24,37,49), (8,24,37,49), (5,8,24,37,49)]
    for subset in knnSets:
        subs = [item - 1 for item in subset]
        subXtrain = exaustiveSearch(Xtrain, subs)
        subXtest = exaustiveSearch(Xtest, subs)
        model = knn.train(subXtrain, ytrain,3)
        yhat = knn.classify(subXtest, model)
        error = err.resub(ytest, yhat)
        print '-> The test error for %s is %f' % (list(subset), error)

    print 'Classification rule: DLDA'
    selectResubExaus(Xtrain, ytrain, 'dlda')
    selectResubForward(Xtrain, ytrain, 'dlda')
    selectLooExaus(Xtrain, ytrain, 'dlda')
    selectLooForward(Xtrain, ytrain, 'dlda')
    dldaSets = [(12,),(57,66),(12,20,33),(12,),(12,20),(12,20,33),(12,16,20,33),(12,16,19,20,33),(12,),(57,66),(2,60,66),(12,),(12,18),(12,18,24),(12,18,23,24),(12,13,18,23,24)]
    for subset in dldaSets:
        subs = [item - 1 for item in subset]
        subXtrain = exaustiveSearch(Xtrain, subs)
        subXtest = exaustiveSearch(Xtest, subs)
        model = gda.ldatrain(subXtrain, ytrain)
        yhat = gda.ldaclassify(model, subXtest)
        error = err.resub(ytest, yhat)
        print '-> The test error for %s is %f' % (list(subset), error)

    print 'Classification rule: SVM (c=0.5)'
    selectResubExaus(Xtrain, ytrain, 'svm')
    selectResubForward(Xtrain, ytrain, 'svm')
    selectLooExaus(Xtrain, ytrain, 'svm')
    selectLooForward(Xtrain, ytrain, 'svm')
    svmSets = [(49,),(60,66),(23,60,66),(49,),(21,49),(5,21,49),(5,6,21,49),(5,6,10,21,49),(49,),(60,66),(3,14,57),(49,),(41,49),(28,41,49),(28,41,49,55),(5,28,41,49,55)]
    for subset in svmSets:
        subs = [item - 1 for item in subset]
        subXtrain = exaustiveSearch(Xtrain, subs)
        subXtest = exaustiveSearch(Xtest, subs)
        model = svm.LinearSVC(C=0.5)
        model.fit(subXtrain, ytrain)
        yhat = model.predict(subXtest)
        error = err.resub(ytest, yhat)
        print '-> The test error for %s is %f' % (list(subset), error)

    # plt.figure(1)
    # plot(subX, ytrain)
    # clf = svm.LinearSVC(C=0.5)
    # clf.fit(subX, ytrain)
    # yhat = clf.predict(subX)
    # model = gda.ldatrain(subX, ytrain)
    # yhat = gda.ldaclassify(model, subX)
    # print err.resub(ytrain, yhat)
    # plt.figure(2)
    # plot(subX, yhat)
    # plt.show()

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
            elif method=='svm':
                model = svm.LinearSVC(C=0.5)
                model.fit(subX, ytrain)
                yhat = model.predict(subX)
            error = err.resub(ytrain, yhat)
            if error < minErr:
                bestSubsets = np.asarray(subsets) + 1
                minErr = error
        print '-> the optimal %d features is %s, with resubstitution %f' % (r, bestSubsets.tolist(), minErr)
    return None

def selectLooExaus(Xtrain, ytrain, method):
    """ Select 1-3 feature using exhaustive search based on leave-one-out error
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
                elif method=='svm':
                    model = svm.LinearSVC(C=0.5)
                    model.fit(modelX, modelY)
                    yhat[i] = model.predict(Xtest)
            error = err.resub(ytrain, yhat)
            if error < minErr:
                bestSubsets = np.asarray(subsets) + 1
                minErr = error
        print '-> the optimal %d features is %s, with leave-one-out error %f' % (r, bestSubsets.tolist(), minErr)
    return None

def selectResubForward(Xtrain, ytrain, method, maxFeatNum=5):
    """ Select 1-5 feature using forward search based on resubstitutions
        of 3NN/DLDA/SVM
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
            elif method=='svm':
                model = svm.LinearSVC(C=0.5)
                model.fit(subX, ytrain)
                yhat = model.predict(subX)
            error = err.resub(ytrain, yhat)
            if error < minErr:
                bestSubsets = np.asarray(subsets)+1
                minErr = error
        featSets = (bestSubsets-1).tolist()
        print '-> the sub-optimal %d features is %s, with rebustitution %f' % (i+1, bestSubsets.tolist(), minErr)
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
                elif method=='svm':
                    model = svm.LinearSVC(C=0.5)
                    model.fit(modelX, modelY)
                    yhat[k] = model.predict(Xtest)
            error = err.resub(ytrain, yhat)
            if error < minErr:
                bestSubsets = np.asarray(subsets)+1
                minErr = error
        featSets = (bestSubsets-1).tolist()
        print '-> the sub-optimal %d features is %s, with leave-one-out error %f' % (i+1, bestSubsets.tolist(), minErr)
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
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    for k in range(K):
        c = np.asarray(clusters[k])
        plt.plot(c[:,0], c[:,1], colors[k])
        # ax.scatter(c[:,0], c[:,1], c[:,2], c=colors[k][1])
    return None

if __name__ == '__main__':
    main()
