#!/usr/bin/env python

import sys
# from numpy import mat, power, mean

def read_input(filename):
    for line in filename:
        yield line.rstrip()

def main():
    input = read_input(sys.stdin)
    mapperOut = [line.split['\t'] for line in input]
    cumVal = 0.0
    cumSumSq = 0.0
    cumN = 0.0
    for ins in mapperOut:
        nj = float(ins[0])
        cumN += nj
        cumVal += nj*float(ins[1])
        cumSumSq += nj*float(ins[2])
    mean = cumVal/cumN
    varSum = (cumSumSq - 2*mean*cumVal + cumN*mean*mean) / cumN
    print '%d\t%f\t%f' % (cumN, mean, varSum)
    print >> sys.stderr, 'report: still alive'