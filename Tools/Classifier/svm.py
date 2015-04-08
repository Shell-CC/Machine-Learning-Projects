#!/usr/bin/env python

"""This is implememtaion of SVM with kernels
"""
import numpy as np

def train(Xtrain, ytrain):
    """Train the weights using SVM

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
    return weights

def classify():
    return
def test():
    return

def simpleSmo(Xtrain, ytrain, N=10000):
    # simpliefied SMO algorithm
    alphas = np.zeros((Xtrain.shape(0)))
    return alphas
def main():
    pass

if __name__ == '__main__':
    main()
