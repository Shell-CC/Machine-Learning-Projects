#!/usr/bin/env python

import sys
from numpy import mat, power, mean

def read_input(filename):
    for line in filename:
        yield line.rstrip()

def main():
    input = read_input(sys.stdin)
    input = [(float)(line) for line in input]
    numInputs = len(input)
    input = mat(input)
    sqInput = power(input, 2)
    print '%d\t%f\t%f' % (numInputs, mean(input), mean(sqInput))
    print >> sys.stderr, 'report: still alive'

if __name__ == '__main__':
    main()
