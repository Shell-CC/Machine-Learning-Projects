#!/usr/bin/env python

import numpy as np

def model(A, B, pi=[]):
    """Get parameters for a Hidden Markov Model.

    parameters
    ----------
    A  : transition matrix, A(i,j) is the probability of transition from state i to state j.
    B  : emission matrix, B(j,k) is the probability of observating k from state j.
    pi : the probability of initial state.

    returns
    -------
    theta: the parameters in matrix theta = (A, B, pi).If unknown, default by uniform distribution
    """
    A = np.mat(A)
    B = np.mat(B)
    if len(pi)==0:
        stateNum = A.shape[0]
        pi = np.ones((stateNum,1))/stateNum
    else:
        pi = np.mat(pi)
    return (A,B,pi)

def likelihood(observ, theta):
    """ Calculate the likelihood of the observed sequence using forward algorithm

    parameters
    ----------
    theta:  (trasition matrix, emission matrix, initial state)
    observ: the observed sequence

    returns
    -------
    likelihood: the probability of the observation
    """
    alpha = forward(theta, observ)
    likelihood = alpha.sum(axis=0)
    return np.asarray(likelihood)[0][0]

def forward(theta, observ):
    """ implement forward algorithm (alpha-recursion)

    parameters
    ----------
    theta:  (trasition matrix, emission matrix, initial state)
    observ: the observed sequence

    returns
    -------
    prob: the joint probability the observation and the present hidden state.
    """
    A = theta[0]
    B = theta[1]
    pi = theta[2]
    t = len(observ)
    observ = np.asarray(observ) - 1
    alpha = np.multiply(pi, B[:,observ[0]])
    for i in range(1,t):
        # print alpha
        alpha = np.multiply(A.T*alpha, B[:,observ[i]])
    return alpha

def main():
    A = [[0.0, 0.3, 0.4, 0.3],
         [0.0, 0.8, 0.1, 0.1],
         [0.0, 0.3, 0.6, 0.1],
         [0.0, 0.1, 0.2, 0.7]]
    B = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.5, 0.2, 0.1, 0.1, 0.05, 0.05],
         [0.05, 0.05, 0.45, 0.05, 0.35, 0.05],
         [0.25, 0.05, 0.05, 0.05, 0.05, 0.55]]
    observ = [3, 2, 1, 2, 2, 6, 5]
    theta = model(A,B)
    for i in theta:
        print i
    print forward(theta, observ)
    print likelihood(observ, theta)

if __name__ == '__main__':
    main()
