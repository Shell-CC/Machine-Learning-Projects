#!/usr/bin/env python
from math import log
import parseData as pd

def train(dataset, featName):
    """Generate a decision tree using ID3

    parameters
    ----------
    dataset: a data set including both features and labels, label is the last item
    label:   name of the features

    returns
    -------
    tree:    the generated decision tree
    """
    tree = ID3(dataset, featName)
    return tree

def classify(X, featName, tree):
    """Classify a list of given data using the decision tree

    parameters
    ----------
    X: a list of data (each data is a list of features) to be classified
    featName: name of the features
    tree:     the decision tree

    returns
    -------
    y:        a list of predicted labels
    """
    yhat = []
    for features in X:
        label = classifyOne(features, featName, tree)
        yhat.append([label])
    return yhat

def test(X, y, featName, tree):
    yhat = classify(X, featName, tree)
    numFeat = len(y)
    error = 0.0
    for i in range(numFeat):
        if not yhat[i]== y[i]:
            error += 1
    errorRate = error/numFeat
    print 'error: %d/%d=%.2f%%' % (error, numFeat, errorRate)
    return yhat

def classifyOne(features, featName, tree):
    firstFeatName = tree.keys()[0]
    branches = tree[firstFeatName]
    axis = featName.index(firstFeatName)
    for key in branches.keys():
        if key == features[axis]:
            subtree = branches[key]
            if type(subtree).__name__ == 'dict':
                label = classifyOne(features, featName, subtree)
            else:
                label = subtree
    return label

def ID3(dataset, featName):
    # generate a decision tree using ID3 algorithm
    labelList = [data[-1] for data in dataset]
    if len(set(labelList)) <= 1:
        return labelList[0]
    else:
        tree = {}
        branches = {}
        # find the best feature (with max info gain) to split
        numFeat = len(dataset[0]) - 1
        bestGain = 0.0
        for i in range(numFeat):
            infoGain = calcInfoGain(i, dataset)
            if (infoGain > bestGain):
                bestGain = infoGain
                bestFeat = i
        tree[featName[bestFeat]] = branches
        # split the data set
        featList = [data[bestFeat] for data in dataset]
        featValues = set(featList)
        for value in featValues:
            subDataset = split(dataset, bestFeat, value)
            branches[value] = ID3(subDataset, featName[:bestFeat]+featName[bestFeat+1:])
    return tree

def entropy(dataset):
    # Calculate entropy of the data set based on number of labels,
    # the last item in each data is the label
    labelVote = {}
    numData = len(dataset)
    for data in dataset:
        label = data[-1]
        labelVote[label] = labelVote.get(label, 0) + 1
    entropy = 0.0
    for key in labelVote:
        prob = float(labelVote[key])/numData
        entropy -= prob * log(prob, 2)
    return entropy

def calcInfoGain(axis, dataset):
    # Calculates the information gain (reduction in entropy) that would
    # result by splitting the data on the chosen feature.
    # axis is the feature column number
    numData = len(dataset)
    featList = [data[axis] for data in dataset]
    featValues = set(featList)
    infoGain = entropy(dataset)
    for value in featValues:
        subDataset = split(dataset, axis, value)
        prob = float(len(subDataset))/numData
        infoGain -= prob * entropy(subDataset)
    return infoGain

def split(dataset, axis, value):
    # split the data on the chosen feature (axis) of the target value (value).
    subDataset = []
    for data in dataset:
        if data[axis] == value:
            subDataset.append(data[:axis]+data[axis+1:])
    return subDataset

def main():
    dataset, none = pd.textAsString('../testData/playGolf', 0, '\t')
    featName = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    testdata, y = pd.textAsString('../testData/playGolf', 5, '\t')
    tree = train(dataset, featName)
    print 'Decision tree:', tree
    test(testdata, y, featName, tree)

if __name__ == '__main__':
        main()
