#!/usr/bin/env python

"""This is the implementation of k-means algorithm
"""

import parseData as pd
from random import sample
import numpy as np

def cluster(X, k):
    X = np.asarray(X)
    cent = kmeans(X, k)
    return cent

def kmeans(X, k):
    numX = X.shape[0]
    # random select k points as clustering centroids
    centroids = sample(X, k)
    # while cluaster changed
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        labels = np.zeros(numX)
        minDist = np.empty(numX)
        minDist.fill(np.Inf)
        for cent in centroids:
            # calculate the distance between the all data and the centroids
            diff = np.tile(cent, (numX,1)) - X
            dist = (diff**2).sum(axis=1)
            minDist = np.minimum(dist, minDist)
            # assign the point to the cluster with lowest distance
            labelChange = dist-minDist
            labelChange[labelChange>0]=-1
            labelChange += 1
            labels += labelChange
        # update the centroids
    return centroids

def main():
    X = pd.textAsFloat('../testData/testKMeans.txt', None, '\t')
    cent = cluster(X, 4)
    print cent

if __name__ == '__main__':
    main()
