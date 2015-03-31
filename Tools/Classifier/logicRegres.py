#!/usr/bin/env python
"""This is the implementation of the logic regression with multiple variables
"""
import numpy as np
import parseData as pd
import matplotlib.pyplot as plt

def train(Xtrain, ytrain, alpha):
    """Train the weights using 'gradient descent'

    parameters
    -----------
    Xtrain: array_like features as train data, x
    ytrain: array_like labels as train data, y
    alpha:  the learning rate (step length) of gradient descent

    Returns
    -------
    weights: weights of features, w
    """
    # change X, y type to mat, add x_0 = 1 to X
    Xtrain = np.mat(Xtrain)
    Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)
    ytrain = np.mat(ytrain)
    # using gradient descent
    weights = gradientDescent(Xtrain, ytrain, alpha);
    return weights

def classify(X, weights):
    """Classify the data X using logic regression using the weights
    yhat = 1 / (1 + exp(-w.*X))

    parameters
    ----------
    X:       array_like features as input data
    weights: weights of the features from train data

    returns
    -------
    yhat:    the predicted labels of the data.
    """
    # change X type to mat, add x_0 = 1 to X
    X = np.mat(X)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    # predict yhat using regression
    yhat = sigmoid(X * weights)
    yhat = np.round(yhat)
    return yhat

def test(Xtest, ytest, weights):
    """Test the logic regression classifier using the data with labels
    yhat = 1 / (1 + exp(-w.*X))

    parameters
    ----------
    Xtest:   array_like features as the input data
    ytest:   array_like labels as the input label
    weights: weights of the features from train data

    returns
    -------
    print:   the misclassificaiton error of the data set
    yhat:    the predicted labels of the data.
    """
    # change y type to mat
    ytest = np.mat(ytest)
    # predict yhat and calaulate misclassification error
    yhat = classify(Xtest, weights)
    error = np.absolute(yhat-ytest).sum()
    errorRate = np.absolute(yhat-ytest).mean()
    print 'error: %d/%d=%.2f%%' % (error, len(yhat), errorRate*100)
    return yhat

def sigmoid(inX):
    return 1/(1+np.exp(-inX))

def gradientDescent(Xtrain, ytrain, alpha=0.001, N=10000, debug=False):
    '''Train the weights using 'gradient descent'
    Gradient descent can converge to a local minimal of the cost function
    The cost function of logic regression is the euclidian distance of Xw and y
    '''

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

def main():
    Xtrain, ytrain = pd.textAsMat('../testData/testLogicRegres', 1, '\t')
    Xtest, ytest = pd.textAsMat('../testData/testLogicRegres2', 1, '\t')
    ytrain = (np.ones((ytrain.shape[0],1)) + ytrain)/2
    ytest = (np.ones((ytest.shape[0],1)) + ytest)/2
    weights = train(Xtrain, ytrain, 0.01);
    test(Xtrain, ytrain, weights)
    test(Xtest, ytest, weights)


if __name__ == '__main__':
    main()
