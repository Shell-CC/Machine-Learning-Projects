# classifier

import numpy as np

def kNN(Xtest, train, k):
    Xtrain = np.asarray(train.keys())
    ytrain = train.values()
    XtestArray = np.tile(Xtest, (Xtrain.shape[0], 1))
    Xdiff = Xtrain - XtestArray
    dist = (Xdiff**2).sum(axis=1)
    labelVote = {}
    maxVote = 0
    for i in dist.argsort()[:k]:
        labelVote[ytrain[i]] = labelVote.get(ytrain[i], 0) + 1
        if labelVote[ytrain[i]] > maxVote:
            ytest = ytrain[i]
            maxVote = labelVote[ytrain[i]]
    return ytest