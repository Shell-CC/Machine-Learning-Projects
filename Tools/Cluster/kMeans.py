#!/usr/bin/env python

"""This is the implementation of k-means algorithm
"""

import parseData as pd
from random import sample
import numpy as np
from matplotlib import pyplot as plt

def cluster(X, k):
    """assign data X into k clusters

    paramters
    ---------
    X: data to be clustered
    k: number of clusters

    returns
    -------
    clust: list of k cluster of data
    """
    X = np.asarray(X)
    y = kmeans(X, k)
    return y

def kmeans(X, k):
    # This algorithm converges to a local minimal
    numX = X.shape[0]
    # random select k points as clustering centroids
    centroids = np.asarray(sample(X, k))
    # while cluster changed
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # assign
        labels = np.zeros(numX)
        minDist = np.empty(numX)
        minDist.fill(np.Inf)
        for i in range(k):
            # calculate the distance between the all data and the centroid i
            diff = np.tile(centroids[i], (numX,1)) - X
            dist = (diff**2).sum(axis=1)
            # assign the point to the cluster with lowest distance
            for j in range(numX):
                if dist[j]<minDist[j]:
                    minDist[j] = dist[j]
                    labels[j] = i
        # update the centroids
        # print centroids
        oldCentroids = centroids.copy()
        clusters = [[] for i in range(k)]
        for i in range(numX):
            clusters[int(labels[i])].append(X[i])
        for i in range(k):
            clusters[i] = np.asarray(clusters[i])
            centroids[i] = clusters[i].mean(axis=0)
            clusters[i] = clusters[i].tolist()
        if not (oldCentroids==centroids).all():
            clusterChanged = True
    return clusters

def main():
    X = pd.textAsFloat('../testData/testKMeans.txt', None, '\t')
    clusters = cluster(X, 4)
    c0 = np.asarray(clusters[0])
    c1 = np.asarray(clusters[1])
    c2 = np.asarray(clusters[2])
    c3 = np.asarray(clusters[3])
    plt.plot(c0[:,0], c0[:,1], 'ob',
             c1[:,0], c1[:,1], 'or',
             c2[:,0], c2[:,1], 'og',
             c3[:,0], c3[:,1], 'oy')
    plt.show()
    # print clusters[1]

if __name__ == '__main__':
    main()
