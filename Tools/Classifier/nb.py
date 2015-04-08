#!/usr/bin/env python

import parseData as pd
import numpy as np

def train(Xtrain, ytrain):
    """Calculate the prior probability and likelihood of the train data

    parameters
    ----------
    Xtrain: input dataset, for each x in (0,1)
    ytrain: input labels

    returns
    -------
    likelihood: P(X=1|Y)
    prior:      P(Y)
    ySet:       the set of all labels
    """
    ySet = set(y[0] for y in ytrain)
    likelihood, prior = calcLikelihoodAndPrior(Xtrain, ytrain, ySet)
    return likelihood, prior, ySet

def classify(Xtest, model):
    """Calculate the posterior probability of the test data, and choose the largest

    parameters
    ----------
    Xtest: the test data in vector, for each x in (0,1)
    model: (likelihood, prior, ySet)

    returns
    -------
    yhatList: a list of predicted label of the test data
    """
    yhatList = []
    for X in Xtest:
        yhat = calcPosterior(X, model[0], model[1], model[2])
        yhatList.append([yhat])
    return yhatList

def test(Xtest, ytest, model):
    """Test the naive Bayes classifier

    parameters
    ----------
    Xtest: the test data in vector, for each x in (0,1)
    model: (likelihood, prior, ySet)

    returns
    -------
    print   : the misclassification error
    yhatList: a list of predicted label of the test data
    """
    yhat = classify(Xtest, model)
    # change y type to array
    ytest = np.mat(ytest)
    print ytest
    yhat = np.mat(yhat)
    print yhat
    error = np.absolute(yhat-ytest).sum()
    errorRate = np.absolute(yhat-ytest).mean()
    print 'error: %d/%d=%.2f%%' % (error, len(yhat), errorRate*100)
    return yhat

def calcLikelihoodAndPrior(Xtrain, ytrain, ySet):
    likelihood = {}
    prior = {}
    N = len(ytrain)
    for y in ySet:
        # in case of the likelihood is 0
        likelihood[y] = np.ones(len(Xtrain[0]))
        prior[y] = 0.0
    for i in range(len(Xtrain)):
        Xmat = np.array(Xtrain[i])
        likelihood[ytrain[i][0]] += Xmat
        prior[ytrain[i][0]] += 1.0
    for key in prior.keys():
        likelihood[key] /= prior[key] + 2
        prior[key] /= N
    return likelihood, prior

def calcPosterior(X, likelihood, prior, ySet):
    maxPost = 0
    for y in ySet:
        post = 1
        for i in range(len(X)):
            if X[i] == 1:
                post *= likelihood[y][i]
            else:
                post *= 1-likelihood[y][i]
        post *= prior[y]
        if post > maxPost:
            maxPost = post
            yhat = y
    return yhat

def createWordList(dataset):
    wordSet = set([])
    for line in dataset:
        wordSet = wordSet | set(line)
    wordList = sorted(list(wordSet))
    return wordList

def words2Vec(dataset, wordList=None):
    if wordList == None:
        wordList = createWordList(dataset)
    wordVecList = []
    for line in dataset:
        wordVec = len(wordList)*[0]
        for word in line:
            if word in wordList:
                wordVec[wordList.index(word)] += 1
            else:
                print 'the word %s is not in' % word
        wordVecList.append(wordVec)
    return wordList, wordVecList

def main():
    dataset, ytrain = pd.textAsString('../testData/wordList.txt', -1)
    testEntry = [['love', 'my', 'delmation'], ['stupid', 'garbage']]
    Xlabel, Xtrain = words2Vec(dataset)
    Xlabel, Xtest = words2Vec(testEntry, Xlabel)
    model = train(Xtrain, ytrain)
    yhat = classify(Xtest, model)
    for i in range(len(yhat)):
        print testEntry[i], yhat[i][0]

if __name__ == '__main__':
    main()
