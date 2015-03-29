# Prepare data for digits recognition.

import os
import numpy as np

def text2tuple(filename, m, n):
    vec = []
    f = open(filename, 'r')
    for i in range(m):
        lineStr = f.readline()
        for j in range(n):
            vec.append((int)(lineStr[j]))
    return tuple(vec)

def makeDict(pathname, m, n):
    items = {};
    for file in os.listdir(pathname):
        if file[-4:] == '.txt':
            filename = pathname + '/' + file
            tup = text2tuple(filename, m, n)
            items[tup] = (int)(file[0])
    return items

def tuple2array(tup, m, n):
    array = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            array[i,j] = tup[i*m+j]
    return array