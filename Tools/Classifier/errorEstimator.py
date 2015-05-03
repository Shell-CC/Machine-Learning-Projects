#!/usr/bin/env python

import numpy as np

def resub(y, yhat, c=2):
    """ cauculate the apparent error (resubstituion) on the trainning set for c=2

    Parameters
    ----------
    y:    labels of the training data
    yhat: predicted values of the classifier

    Returns
    -------
    errRate: errors committed / number of points
    """
    if c==2:
        errRate = np.absolute(y - yhat).mean()
    return errRate

def main():
    pass

if __name__ == '__main__':
    main()
