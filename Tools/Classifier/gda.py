#!/usr/bin/env python

import numpy as np

def param(X, y, equal=True):
    """return the mus and sigmas of the sample data
    """
    # group with label
    labelNum = len(set(y))
    (N, D) = X.shape
    Xs = [[] for _ in range(labelNum)]
    for i in range(len(y)):
        Xs[y[i]].append(X[i])
    for i in range(labelNum):
        Xs[i] = np.asarray(Xs[i])
    # Cauclulate p
    if equal:
        probs = np.ones(labelNum)/labelNum
    else:
        probs = np.zeros(labelNum)
        for i in range(labelNum):
            probs[i] = (Xs[i].shape[0])/float(N)
    # Calculate mu
    mus = np.zeros((labelNum, D))
    for i in range(labelNum):
        mus[i] = Xs[i].mean(axis=0)
    # Calculate sigmas
    sigmas = []
    Ns = np.empty(labelNum, dtype=int)
    for i in range(labelNum):
        sigmas.append(np.zeros((D,D)))
        Ns[i] = Xs[i].shape[0]
        for n in range(Ns[i]):
            diff = np.mat(Xs[i][n] - mus[i])
            sigmas[i] += diff.T * diff
        sigmas[i] = np.asarray(sigmas[i]) / (Ns[i]-1)
    return mus, sigmas, probs, Ns

def ldatrain(Xtrain, ytrain):
    (mus, sigmas, probs, Ns) = param(Xtrain, ytrain)
    D = Xtrain.shape[1]
    sigma = np.eye(D)
    for i in range(D):
        sigma[i,i] = (Ns[0]-1)*sigmas[0][i,i]+(Ns[1]-1)*sigmas[1][i,i]
        sigma[i,i] = sigma[i,i]/(Ns[0]+Ns[1]-2)
    # train
    sigmaInv = np.mat(np.linalg.inv(sigma))
    muDiff = np.mat(mus[1]-mus[0])
    a = (sigmaInv * muDiff.T).T
    a = np.asarray(a)[0]
    b = -0.5*muDiff*sigmaInv*(np.mat(mus[1]+mus[0]).T)
    b = np.asarray(b)[0][0]
    return (a,b)

def ldaclassify(param, Xtest):
    a = np.mat(param[0])
    y = np.mat(Xtest) * a.T
    y = np.asarray(y.T)[0] + param[1]
    y[y<0]=0
    y[y>0]=1
    yhat = map(int, y)
    return np.asarray(yhat)

def main():
    pass

if __name__ == '__main__':
    main()
