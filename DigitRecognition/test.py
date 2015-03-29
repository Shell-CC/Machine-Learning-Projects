# test.py

import classify as cls
import parseData as pd

def testkNN():
    train = pd.makeDict('./trainDigits', 32, 32)
    test = pd.makeDict('./testDigits', 32, 32)
    ydiff = []
    error = 0.0
    for Xtest in test.keys():
        yhat = cls.kNN(Xtest, train, 1)
        ydiff.append(yhat == test[Xtest])
        if not yhat == test[Xtest]:
            error += 1.0
    errorRate = error/len(ydiff)
    return ydiff, errorRate