#!/usr/bin/env python

import numpy as np

def main():
    (Xtrain, ytrain, IDtrain, names) = readData('Training_Data.txt')
    print Xtrain
    print ytrain
    print exaustiveSearch(Xtrain, [0,1])
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
    print 'Read %d datas from %s.' % (len(IDs), filename)
    print 'Each data contains %d genes, with a ID and a label.' % len(names)
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
