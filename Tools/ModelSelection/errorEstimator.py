#!/usr/bin/env python

import numpy as np
from sklearn.cross_validation import LeaveOneOut

def resubstitution(clf, Xtrain, ytrain):
    """ cauculate the apparent error (resubstituion) on the trainning set for c=2

    Parameters
    ----------
    y:    labels of the training data
    yhat: predicted values of the classifier

    Returns
    -------
    errRate: errors committed / number of points
    """
    clf.fit(Xtrain, ytrain)
    yhat = clf.predict(Xtrain)
    mse = ((yhat-ytrain)**2).mean()
    rmse = mse**0.5
    return rmse

def leaveOneOut(clf, Xtrain, ytrain):
    n = len(ytrain)
    loo = LeaveOneOut(n)
    rmses = []
    for train, test in loo:
        xtr = np.take(Xtrain, train, axis=0); ytr = np.take(ytrain, train)
        # print xtr.shape, ytr.shape
        xte = np.take(Xtrain, test, axis=0); yte = np.take(ytrain, test)
        clf.fit(xtr, ytr)
        yhat = clf.predict(xte)
        rmses.append(abs((yhat-yte)[0]))
    rmses = np.asarray(rmses)
    return rmses.mean()

def main():
    pass

if __name__ == '__main__':
    main()
