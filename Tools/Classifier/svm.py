#!/usr/bin/env python

"""This is implememtaion of SVM with kernels
"""
import parseData as pd
import numpy as np
from random import randint

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
    # Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)
    ytrain = np.mat(ytrain)
    alphaList, b = simpleSMO(Xtrain, ytrain, 0.6, 0.0001, 40)
    return alphaList, b

def classify():
    return
def test():
    return

def simpleSMO(Xtrain, ytrain, C, toler, N=10000):
    # simpliefied SMO algorithm for training SVM
    numAlphas = Xtrain.shape[0]
    alphaList = np.mat(np.zeros((numAlphas, 1)))
    b = 0
    iter = 0
    while (iter < N):
        alphaPairsChanged = 0
        # select alpha i and alpha j
        for i in range(numAlphas):
            # check if can be optimized, yes if error is larger than toler
            fXi = np.multiply(alphaList, ytrain).T * (Xtrain*Xtrain[i,:].T) + b
            Ei = fXi - ytrain[i]
            if ((ytrain[i]*Ei<-toler) and (alphaList[i]<C)) or \
               ((ytrain[i]*Ei>toler) and (alphaList[i]>0)):
                j = selectJ(i, numAlphas)
                fXj = np.multiply(alphaList, ytrain).T * (Xtrain*Xtrain[j,:].T) + b
                Ej = fXj - ytrain[j]
                # optimize alpha i and alpha j
                # garantee that alpha j between 0 and C
                if (ytrain[i] == ytrain[j]):
                    L = max(0, alphaList[i]+alphaList[j]-C)
                    H = min(C, alphaList[i]+alphaList[j])
                else:
                    L = max(0, alphaList[j]-alphaList[i])
                    H = min(C, alphaList[j]-alphaList[i]+C)
                if L==H:
                    # print 'L=H'
                    continue
                eta = 2.0*Xtrain[i,:]*Xtrain[j,:].T - \
                      Xtrain[i,:]*Xtrain[i,:].T - \
                      Xtrain[j,:]*Xtrain[j,:].T
                if eta>=0:
                    continue
                alphaJold = alphaList[j].copy()
                alphaIold = alphaList[i].copy()
                # update alpha j
                alphaList[j] -= ytrain[j]*(Ej- Ei) / eta
                if alphaList[j]>H:
                    alphaList[j] = H
                elif alphaList[j]<L:
                    alphaList[j] = L
                if (abs(alphaList[j]-alphaJold)<0.00005):
                    continue
                # update alpha i
                alphaList[i] += ytrain[i]*ytrain[j]*(alphaIold-alphaList[i])
                # compute the b threshold
                b1 = b - Ei- \
                     ytrain[i]*(alphaList[i]-alphaIold)*Xtrain[i,:]*Xtrain[i,:].T\
                     - ytrain[j]*(alphaList[j]-alphaJold)*Xtrain[j,:]*Xtrain[j,:].T
                b2 = b - Ej- \
                     ytrain[i]*(alphaList[i]-alphaIold)*Xtrain[i,:]*Xtrain[i,:].T\
                     - ytrain[j]*(alphaList[j]-alphaJold)*Xtrain[j,:]*Xtrain[j,:].T
                if (alphaList[i]>0) and (alphaList[i]<C):
                    b = b1
                elif (alphaList[j]>0) and (alphaList[j]<C):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaPairsChanged += 1
        if alphaPairsChanged==0:
            iter += 1
        else:
            iter = 0
        print 'iter number: %d' % iter
    return alphaList, b

def selectJ(i, m):
    # randomly select alpha j not equal to alpha i
    j = i
    while (j==i):
        j = randint(0,m)
    return j

def main():
    # Xtrain, ytrain = pd.textAsFloat('../testData/bclass-train', 0, '\t')
    # Xtest, ytest = pd.textAsFloat('../testData/bclass-test', 0, '\t')
    Xtrain, ytrain = pd.textAsFloat('../testData/testSMO.txt', -1, '\t')
    alphas, b = train(Xtrain, ytrain)
    print b
    print alphas

if __name__ == '__main__':
    main()
