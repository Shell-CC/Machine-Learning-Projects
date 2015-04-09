#!/usr/bin/env python

"""This is the implementation of k-means algorithm
"""

import parseData as pd

def main():
    X = pd.textAsFloat('../testData/testKMeans.txt', None, '\t')
    print X

if __name__ == '__main__':
    main()
