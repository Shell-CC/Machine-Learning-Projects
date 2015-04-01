#!/usr/bin/env python
"""This is the implementation of the k-Nearest Neighbor classier
"""
import numpy as np
import parseData as pd

def train(Xtrain, ytrain, k=3):
    """Set all the data in the train set as landmark points

    parameters
    -----------
    Xtrain: array_like features as train data, x
    ytrain: array_like labels as train data, y
    k:     number of neighbors to be compared

    Returns
    -------
    (Xtrain, ytrain, k)
    """
    Xtrain = np.asarray(Xtrain)
    ytrain = np.asarray(ytrain)
    return (Xtrain, ytrain, k)

def classify(X, model):
    """Classify the data X compared with the train data

    parameters
    ----------
    X:     array_like features as input data
    model: train data and labels
    k:     number of neighbors to be compared

    returns
    -------
    yhat:  the predicted labels of the data.
    """
    n = len(X)
    yhat = np.zeros((n,1))
    for i in range(n):
        yhat[i] = kNN(model[0], model[1], X[i], model[2])
    return yhat

def test(Xtest, ytest, model):
    """Test the kNN classifier using the data with labels

    parameters
    ----------
    Xtest: array_like features as the input data
    ytest: array_like labels as the input label
    model: train data and labels

    returns
    -------
    print: the misclassificaiton error of the data set
    yhat:  the predicted labels of the data
    """
    # change y type to array
    ytest = np.asarray(ytest)
    yhat = classify(Xtest, model)
    error = np.absolute(yhat-ytest).sum()
    errorRate = np.absolute(yhat-ytest).mean()
    print 'error: %d/%d=%.2f%%' % (error, len(yhat), errorRate*100)
    return yhat

def kNN(Xtrain, ytrain, Xtest, k):
    """Given a new data, find the label,
    which is the k nearest neighbors in the trainning set
    """
    ytrain = np.transpose(ytrain)[0]
    # calculate the eclidian distacne between the test data and all train data
    XDiff = np.tile(Xtest, (Xtrain.shape[0], 1)) - Xtrain
    dist = (XDiff**2).sum(axis=1)
    # vote for the k nearset neighbors, and choose the max vote as the label
    labelVote = {}
    maxVote = 0
    for i in dist.argsort()[:k]:
        labelVote[ytrain[i]] = labelVote.get(ytrain[i], 0) + 1
        if labelVote[ytrain[i]] > maxVote:
            yhat = ytrain[i]
            maxVote = labelVote[ytrain[i]]
    return yhat

def main():
    Xtrain, ytrain = pd.textAsFloat('../testData/bclass-train', 1, '\t')
    Xtest, ytest = pd.textAsFloat('../testData/bclass-test', 1, '\t')
    model = train(Xtrain, ytrain, 3)
    test(Xtrain, ytrain, model)
    test(Xtest, ytest, model)

if __name__ == '__main__':
    main()