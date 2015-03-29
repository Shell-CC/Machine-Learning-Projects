import numpy as np
# import operator

def createDataSet():
    group = np.array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def kNNClassify(Xtest, Xtrain, ytrain, k):
    trainSize = Xtrain.shape[0]
    XDiff = np.tile(Xtest, (trainSize, 1)) - Xtrain
    sqXDiff = XDiff**2
    sqDist = sqXDiff.sum(axis=1)
    dist = sqDist**0.5
    sortIndex = dist.argsort()
    classDict = {}
    maxvalue = -1
    for i in sortIndex[:k]:
        classDict[ytrain[i]] = classDict.get(ytrain[i], 0) + 1
        if classDict[ytrain[i]] > maxvalue:
            label = ytrain[i]
            maxvalue = classDict[ytrain[i]]
    return label
