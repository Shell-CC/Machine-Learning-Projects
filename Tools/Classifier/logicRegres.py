#!/usr/bin/env python

'''This is the implementation of the linear regression with multiple variables
'''
import numpy as np
import parseData as pd
import matplotlib.pyplot as plt

def sigmoid(inX):
    return 1/(1+np.exp(-inX))

def gradientDescentTrain(Xtrain, ytrain, alpha=0.001, N = 3000, debug=False):
    '''Train the weights using 'gradient descent'
    Gradient descent can converge to a local minimal of the cost function
    The cost function of logic regression is the euclidian distance of Xw and y

    parameters
    -----------
    Xtrain: array_like features as train data, x
    ytrain: array_like labels as train data, y
    alpha:  the learning rate (step length), the default is 0.001
    N:      the number of iterations

    Returns
    -------
    weights: weights of features, w
    '''

    # change type to mat, add x_0 = 1
    Xtrain = np.mat(Xtrain)
    Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)
    ytrain = np.mat(ytrain)
    # start with the weights all set to 1
    weights = np.ones((Xtrain.shape[1],1))
    # calculate gradians N times
    alpha = 0.001
    converg = []
    for k in range(N):
        h = sigmoid(Xtrain*weights)
        j = -(np.multiply(ytrain, np.log(h)) + np.multiply((1-ytrain), np.log(1-h))).mean()
        converg.append(j)
        ydiff = h - ytrain
        weights = weights - alpha*Xtrain.T*ydiff
    if debug:
        plt.plot(range(N), converg)
        plt.show()
    return weights

def classify(X, weights):
    # change type to mat, add x_0 = 1
    X = np.mat(X)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    yhat = sigmoid(X * weights)
    yhat = np.round(yhat)
    return yhat

def test(Xtest, ytest, weights):
    ytest = np.mat(ytest)
    yhat = classify(Xtest, weights)
    error = np.absolute(yhat-ytest).sum()
    errorRate = np.absolute(yhat-ytest).mean()
    return yhat, error, errorRate

def main():
    Xtrain, ytrain = pd.textAsMat('../testData/testLogicRegres', 1, '\t')
    Xtest, ytest = pd.textAsMat('../testData/testLogicRegres2', 1, '\t')
    ytrain = (np.ones((ytrain.shape[0],1)) + ytrain)/2
    ytest = (np.ones((ytest.shape[0],1)) + ytest)/2
    weights = gradientDescentTrain(Xtrain, ytrain, 0.01, 300000)
    yhat, error, errorRate = test(Xtrain, ytrain, weights)
    print error, errorRate
    yhat, error, errorRate = test(Xtest, ytest, weights)
    print error, errorRate


if __name__ == '__main__':
    main()
