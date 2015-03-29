#!/usr/bin/env python

'''This is the implementation of the linear regression with multiple variables
'''
import numpy as np

def normalEquationTrain(Xtrain, ytrain):
    ''' Train the parameters using 'Normal Equation'
    y = w0 + w1*x1 + w2*x2...

    Parameters
    ----------
    Xtrain: array_like features as train data, x
    ytrain: array_like labels as train data, y

    Returns
    -------
    weights: weights of features, w
    '''
    # change type to mat, add x_0 = 1
    Xtrain = np.mat(Xtrain)
    Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)
    ytrain = np.mat(ytrain)
    # calculate w = (xT * x)-1 * xT * y
    xTx = Xtrain.T * Xtrain
    if np.linalg.det(xTx) == 0:
        # should use ridge regression here
        print 'The train data has reduntant features'
        return
    weights = xTx.I * Xtrain.T * ytrain
    return weights

def predict(weights, X):
    X = np.mat(X)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    yhat = X * weights
    return yhat

def predictTest(weights, X, y):
    yhat = predict(weights, X)
    corr = np.corrcoef(yhat.T, np.mat(y).T)
    return yhat, corr[0][1]

def main():
    pass

    if __name__ == '__main__':
        main()