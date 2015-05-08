#!/usr/bin/env python
import numpy as np

def numEncoder(Xtrain, Xtest, mask='all'):
    X = Xtrain + Xtest
    if mask=='all':
        mask = range(len(X[0]))
    newFeat = []
    encoder = [{} for _ in xrange(len(X[0]))]
    for i in mask:
        valueSet = sorted(list(set([x[i] for x in X])))
        count = 0
        for value in valueSet:
            newFeat.append(value)
            encoder[i][value] = count
            count += 1
    for x in Xtrain:
        for i in mask:
            x[i] = encoder[i].get(x[i])
    for x in Xtest:
        for i in mask:
            x[i] = encoder[i].get(x[i])
    return encoder

def main():
    pass

if __name__ == '__main__':
    main()
