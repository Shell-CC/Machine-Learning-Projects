#!/usr/bin/env python

import kMeans
import parseData as pd
import numpy as np

def main():
    X = pd.textAsFloat('../testData/testCluster', None, ' ')
    X = np.asarray(X)
    print X.shape
    # clusters = kMeans.cluster(X, 2)
    # print clusters

if __name__ == '__main__':
    main()
