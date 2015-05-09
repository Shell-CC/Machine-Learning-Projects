#!/usr/bin/env python

import parseData as pd
from collections import Counter
import csv
import numpy as np
import matplotlib.pyplot as plt
import Visualization.show as show
from Preprocess.encoder import numEncoder
import ModelSelection.errorEstimator as err
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.cross_validation import LeaveOneOut

def main():
    Xtrain, ytrain = pd.csvAsString('train.csv', -1)
    Xtest = pd.csvAsString('test.csv')
    # print 'For the train data, ',
    years, cities, groups, types, P = analyData(Xtrain, False)
    # print 'For the test data, ',
    years2, cities2, groups2, types2, P2 = analyData(Xtest, False)
    # for i in xrange(37):
        # P[i] = P[i]+P2[i]
    # rvDomain(years+years2, cities+cities2, groups+groups2, types+types2, P, True)
    # rvDistrib(years+years2, cities+cities2, groups+groups2, types+types2, P, True)

    kNNReg(Xtrain, ytrain, Xtest, 'kNNRegHalf.csv', [0,1,2,3,4,5])
    kNNReg(Xtrain, ytrain, Xtest, 'kNNRegAll.csv', 'all')

    treeReg(Xtrain, ytrain, Xtest, 'treeReg.csv')

    cpp(Xtrain, ytrain, Xtest, 'cpp_tree.csv', DecisionTreeClassifier(),5)

    chooseClassify(Xtrain, ytrain, Xtest)

def chooseClassify(Xtrain, ytrain, Xtest):
    encoder = numEncoder(Xtrain, Xtest, 'all')
    Xtrain = np.asarray(Xtrain)
    ytrain = np.asarray(ytrain)
    Xtest = np.asarray(Xtest)
    clfs = [DecisionTreeClassifier(criterion='gini'),
            DecisionTreeClassifier(criterion='entropy')]
    for clf in clfs:
        for k in range(2, 15):
            clu = KMeans(n_clusters=k)
            loo = LOO_CCP(clu, clf, Xtrain, ytrain)[1]
            print loo
    # chooseKeans(Xtrain, ytrain, Xtest, clf)
    return None

def cpp(Xtrain, ytrain, Xtest, filename, clf, k):
    encoder = numEncoder(Xtrain, Xtest, 'all')
    Xtrain = np.asarray(Xtrain)
    ytrain = np.asarray(ytrain)
    Xtest = np.asarray(Xtest)
    # chooseKeans(Xtrain, ytrain, Xtest)
    # cluster
    clu = KMeans(n_clusters=k)
    labels = np.mat(ytrain).T
    clu.fit(labels)
    centers = clu.cluster_centers_.tolist()
    ctrain = clu.predict(labels)
    # classify
    clf.fit(Xtrain, ctrain)
    chat = clf.predict(Xtest)
    # predict
    yhat = clusterPredict(chat, centers)
    write(yhat, filename)
    return None

def chooseKeans(Xtrain, ytrain, Xtest, clf):
    loos = []; resubs = []
    for k in range(2,15):
        clu = KMeans(n_clusters=k)
        loo = LOO_CCP(clu, clf, Xtrain, ytrain)
        resub = Resub_CCP(clu, clf, Xtrain, ytrain)
        print k, resub, loo
        loos.append(loo)
        resubs.append(resub)
    loos = np.asarray(loos); resubs = np.asarray(resubs)
    plt.figure(1)
    plt.plot(range(2,15), loos[:, 0], c='b', label='leave-one-out')
    plt.plot(range(2,15), resubs[:, 0], c='r', label='resubstitution')
    plt.title('Estimate error of Cluster-Classify-Predict')
    plt.xlabel('k')
    plt.ylabel('Validation MSE')
    plt.legend(loc=3)
    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(range(2,15), loos[:, 1], 'ob', label='leave-one-out')
    plt.plot(range(2,15), resubs[:, 1], 'or', label='resubstitution')
    plt.title('Estimate classification error of Cluster-Classify-Predict')
    plt.xlabel('k')
    plt.ylabel('Validation MSE')
    plt.legend(loc=4)
    plt.subplot(2,1,2)
    plt.plot(range(2,15), loos[:, 2], 'b', label='leave-one-out')
    plt.plot(range(2,15), resubs[:, 2], 'r', label='resubstitution')
    plt.title('Estimate clustering error of Cluster-Classify-Predict')
    plt.xlabel('k')
    plt.ylabel('Validation MSE')
    plt.legend(loc=1)
    plt.show()
    return None

def Resub_CCP(clu, clf, Xtrain, ytrain):
    # cluster
    labels = np.mat(ytrain).T
    clu.fit(labels)
    centers = clu.cluster_centers_.tolist()
    ctrain = clu.predict(labels)
    yhat = clusterPredict(ctrain, centers)
    cluErr = ((ytrain-np.asarray(yhat))**2).mean()**0.5
    # classify
    clf.fit(Xtrain, ctrain)
    chat = clf.predict(Xtrain)
    # predict
    yhat = clusterPredict(chat, centers)
    # error
    rmse = ((ytrain-np.asarray(yhat))**2).mean()**0.5
    cdiff = ctrain-chat
    return rmse, (cdiff!=0).sum(), cluErr

def LOO_CCP(clu, clf, Xtrain, ytrain, show=False):
    n = len(ytrain)
    loo = LeaveOneOut(n)
    labels = np.mat(ytrain).T
    rmses = []; cdiffs = 0; cluErrs = []
    for train, test in loo:
        # cluster
        xtr = np.take(Xtrain, train, axis=0); ytr = np.take(labels, train, axis=0)
        xte = np.take(Xtrain, test, axis=0); yte = np.take(labels, test, axis=0)[0]
        clu.fit(ytr)
        ctr = clu.predict(ytr); cte =clu.predict(yte)
        centers = clu.cluster_centers_.tolist()
        cluErr = abs(clusterPredict(cte, centers)[0] - yte[0,0])
        # classify
        clf.fit(xtr, ctr)
        chat = clf.predict(xte)
        if chat!=cte:
            cdiffs += 1
        # predict
        yhat = clusterPredict(chat, centers)[0]
        if show:
            plotLooCpp(ytr, ctr, centers, yte[0,0], cte[0], chat, yhat)
            plt.show()
            show = False
        # error
        rmses.append(abs((yhat-yte[0,0])))
        cluErrs.append(cluErr)
    rmses = np.asarray(rmses); cluErrs = np.asarray(cluErrs)
    return rmses.mean(), cdiffs/float(n), cluErrs.mean()

def plotLooCpp(ytr, ctr, centers, ytest, ctest, chat, yhat):
    N = len(ytr)
    colors = ['b', 'r', 'g', 'c', 'm', 'b', 'r', 'g', 'c', 'm']
    # raw data
    plt.subplot(2,2,1)
    plt.plot(np.asarray(ytr[:, 0]), 'ob')
    plt.plot(N/2, ytest, 'xk')
    plt.title('raw data')

    plt.subplot(2,2,2)
    show.plot1DwithLabel(ytr, ctr)
    for i in range(len(centers)):
        plt.plot(range(N), [centers[i] for _ in range(N)], colors[i])
    color = '%sx' % colors[ctest]
    plt.plot(N/2, ytest, color)
    plt.title('Clustering')

    plt.subplot(2,2,3)
    for i in range(len(centers)):
        plt.plot(range(N), [centers[i] for _ in range(N)], colors[i])
    color = '%sx' % colors[chat]
    plt.plot(N/2, ytest, color)
    plt.title('Classification')

    plt.subplot(2,2,4)
    for i in range(len(centers)):
        plt.plot(range(N), [centers[i] for _ in range(N)], 'w')
    plt.plot(N/2, yhat, color)
    plt.plot(N/2, ytest, 'xk')
    plt.title('prediction')
    return None

def clusterMSE(clu, ytrain):
    labels = np.mat(ytrain).T
    clu.fit(labels)
    c = clu.labels_
    yhat = clusterPredict(c, clu.cluster_centers_.tolist())
    rmse = ((np.asarray(ytrain)-np.asarray(yhat))**2).mean()**0.5
    return rmse

def clusterPredict(labels, centers):
    yhat = []
    for c in labels:
        yhat.append(centers[c][0])
    return yhat

def treeReg(Xtrain, ytrain, Xtest, filename):
    encoder = numEncoder(Xtrain, Xtest, 'all')
    Xtrain = np.asarray(Xtrain)
    ytrain = np.asarray(ytrain)
    Xtest = np.asarray(Xtest)
    d = chooseDepth(Xtrain, ytrain, True)
    tree = DecisionTreeRegressor(max_depth=d)
    tree.fit(Xtrain, ytrain)
    yhat = tree.predict(Xtrain)
    plt.plot(ytrain, 'ro', label='labels')
    plt.plot(yhat, 'b', label='predictions')
    plt.legend()
    plt.show()
    yhat = tree.predict(Xtest)
    print len(set(yhat))
    write(yhat, filename)
    return None

def chooseDepth(Xtrain, ytrain, show=False):
    loos = []
    for d in range(1,20):
        tree = DecisionTreeRegressor(max_depth=d)
        loos.append(err.leaveOneOut(tree, Xtrain, ytrain))
    bestDepth = np.argmin(np.asarray(loos))
    if show:
        print 'the best depth is %d, with %f' % (bestDepth+1, loos[bestDepth])
        plt.plot(range(1,20), loos, c='r', label='leave-one-out cross-validation')
        plt.title('Validation of decision tree of different depth')
        plt.xlabel('k')
        plt.ylabel('RMSE')
        plt.legend(loc=4)
        plt.show()
    return bestDepth+1

def kNNReg(Xtrain, ytrain, Xtest, filename, encoder):
    Xtrain, ytrain, Xtest = encoding(Xtrain, ytrain, Xtest, encoder)
    k = chooseK(Xtrain, ytrain, True)
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(Xtrain, ytrain)
    yhat = knn.predict(Xtrain)
    plt.plot(ytrain, 'ro', label='labels')
    plt.plot(yhat, 'b', label='predictions')
    plt.legend()
    plt.show()
    yhat = knn.predict(Xtest)
    write(yhat, filename)
    return None

def chooseK(Xtrain, ytrain, show=False):
    resubs = []; loos = [];
    for k in range(1,11):
        knn = KNeighborsRegressor(n_neighbors=k)
        resubs.append(err.resubstitution(knn, Xtrain, ytrain))
        loos.append(err.leaveOneOut(knn, Xtrain, ytrain))
    bestK = np.argmin(np.asarray(loos)) + 1
    if show:
        print 'the best k is %d, with %f' % (bestK, loos[bestK-1])
        plt.plot(range(1,11), resubs, c='b', label='resubstituion')
        plt.plot(range(1,11), loos, c='r', label='leave-one-out cross-validation')
        plt.title('Validation of kNN regression of different k')
        plt.xlabel('k')
        plt.ylabel('RMSE')
        plt.legend(loc=4)
        plt.show()
    return bestK

def encoding(Xtrain, ytrain, Xtest, mask='all'):
    """ Encode all
    """
    encoder = numEncoder(Xtrain, Xtest, mask)
    enc = OneHotEncoder(categorical_features=mask)
    enc.fit(Xtrain+Xtest)
    Xtrain = enc.transform(Xtrain).toarray()
    Xtest = enc.transform(Xtest).toarray()
    ytrain = np.asarray(ytrain)
    return Xtrain, ytrain, Xtest


def analyData(X, show=True, y=None):
    num = len(X)
    days = []; months = []; years = []; cities = []; groups = []
    types = []; P = [[] for _ in xrange(37)]
    for x in X:
        x.insert(1, int(x[0][0:2]))
        x.insert(2, int(x[0][3:5]))
        x.insert(3, int(x[0][6:]))
        x.pop(0)
        months.append(int(x[0]))
        days.append(int(x[1]))
        years.append(int(x[2]))
        cities.append(x[3])
        groups.append(x[4])
        types.append(x[5])
        for i in xrange(37):
            x[i+6] = float(x[i+6])
            P[i].append(x[i+6])
    if show:
        print 'there are %d data in this set' % num,
        print 'each has %d features' % len(X[0])
        print '-> There are %d days' % len(set(days))
        print '-> There are %d months' % len(set(months))
        print '-> There are %d years' % len(set(years))
        print '-> There are %d cities' % len(set(cities))
        print '-> There are %d city groups' % len(set(groups))
        print '-> There are %d types of restuarant' % len(set(types))
        for i in xrange(37):
            print '-> For obfuscated data, P%d has %d values;' % (i+1, len(set(P[i])))
    return years, cities, groups, types, P

def rvDomain(years, cities, groups, types, P, show=True):
    yearDomain = list(set(years))
    cityDomain = list(set(cities))
    groupDomain = list(set(groups))
    typeDomain = list(set(types))
    # P = map(float, P)
    PDomain = []
    for i in xrange(37):
        PDomain.append(list(set(P[i])))
    if show:
        print '-> Years: %s' % sorted(yearDomain)
        print '-> Cities: %d %s' % (len(cityDomain), cityDomain)
        print '-> Groups: %s' % groupDomain
        print '-> Types: %s' % typeDomain
        for i in xrange(37):
            print '-> P%d: %s' % (i+1, sorted(PDomain[i]))
    return yearDomain, cityDomain, groupDomain, typeDomain, PDomain

def rvDistrib(years, cities, groups, types, P, write=True):
    yearDis = Counter(years)
    cityDis = Counter(cities)
    groupDis = Counter(groups)
    typeDis = Counter(types)
    PDis = []
    for i in xrange(37):
        PDis.append(Counter(P[i]))
    if write:
        with open('distribution.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(yearDis.keys())
            writer.writerow(yearDis.values())
    return None
def printCounter(c):
    for item in sorted(c.keys()):
        print item,'\t',
    print
    for item in sorted(c.keys()):
        print c[item],'\t',
    print

def write(y, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Prediction'])
        for i in xrange(len(y)):
            writer.writerow([i, y[i]])
    return None

if __name__ == '__main__':
    main()