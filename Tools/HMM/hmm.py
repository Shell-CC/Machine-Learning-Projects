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
    theta: the parameters in array theta = (A, B, pi).If unknown, default by uniform distribution
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if len(pi)==0:
        stateNum = A.shape[0]
        pi = np.ones(stateNum)/stateNum
    else:
        pi = np.asarray(pi)
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
    alpha = forward(theta, observ)[-1]
    likelihood = alpha.sum()
    return likelihood

def filtering(observ, theta):
    """ Infer the present hidden state

    parameters
    ----------
    theta:  (trasition matrix, emission matrix, initial state)
    observ: the observed sequence

    returns
    -------
    present: the conditional probability of the present hidden state given observation
    """
    alpha = forward(theta, observ)[-1]
    present = alpha / alpha.sum()
    return present

def smoothing(observ, theta, t):
    """ Infer the past hidden state at time t

    parameters
    ----------
    theta:  (trasition matrix, emission matrix, initial state)
    observ: the observed sequence
    t:      time t in the sequence,  0<=t<=total time

    returns
    -------
    present: the conditional probability of the hidden state at t given observation
    """
    gammas = forwardBackward(theta, observ, t)
    return gammas[t]

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
    alphas = np.zeros((t, A.shape[0]))
    alphas[0] = pi * B[:,observ[0]]
    for i in range(1,t):
        alphas[i] = np.dot(alphas[i-1], A) * B[:,observ[i]]
    return alphas

def backward(theta, observ, t=0):
    """ implement forward algorithm (alpha-recursion)

    parameters
    ----------
    theta:  (trasition matrix, emission matrix, initial state)
    observ: the observed sequence
    t:      time t in the sequence,  0<=t<=total time

    returns
    -------
    prob: the joint probability the observation and the present hidden state.
    """
    A = theta[0]
    B = theta[1]
    total = len(observ)
    observ = np.asarray(observ) - 1
    betas = np.zeros((total, A.shape[0]))
    betas[total-1] = np.ones(A.shape[0])
    for i in range(total-2,t-1,-1):
        # print np.transpose(A)
        betas[i] = np.dot(betas[i+1]*B[:,observ[i+1]], np.transpose(A))
    return betas

def forwardBackward(theta, observ, t=0):
    """ implememt forward-backward algorithm

    parameters
    ----------
    theta:  (trasition matrix, emission matrix, initial state)
    observ: the observed sequence
    t:      time t in the sequence,  0<=t<=total time

    returns
    -------
    gammas: the conditional probability distribution of hidden path after t.
    """
    alphas = forward(theta, observ)
    likeli = alphas[-1].sum()
    betas = backward(theta, observ,t)
    gammas = alphas * betas / likeli
    return gammas

def viterbi(theta, observ):
    """ implement Viterbi algorithm
    parameters
    ----------
    theta:  (trasition matrix, emission matrix, initial state)
    observ: the observed sequence

    returns
    -------
    omegas: the most likely hidden path list
    """
    A = theta[0]
    B = theta[1]
    pi = theta[2]
    t = len(observ)
    observ = np.asarray(observ) - 1

    argmaxMu = np.zeros((t, A.shape[0]))
    delta = pi * B[:, observ[0]]
    for i in range(1,t):
        mu = np.transpose(A) * np.tile(delta, (3,1))
        delta = B[:,observ[i]] * np.amax(mu, axis=1)
        argmaxMu[i] = np.argmax(mu, axis=1)

    omegas = [0]*t
    omegas[t-1] = (int)(np.argmax(delta))
    for i in range(t-2, -1, -1):
        omegas[i] = (int)(argmaxMu[i+1][omegas[i+1]])
        omegas[i+1] += 1
    return omegas

def greedy(theta, observ, t=None):
    """ Find the most likely hidden path using greedy algorithm,
        by finding the most likely hidden state each time.
        This is not the most likely hidden path, not consist with Viterbi
    """
    alphas = forward(theta, observ)
    likeli = alphas[-1].sum()
    betas = backward(theta, observ,t)
    gammas = alphas * betas / likeli
    return gammas

def main():
    # A = [[0.0, 0.3, 0.4, 0.3],
         # [0.0, 0.8, 0.1, 0.1],
         # [0.0, 0.3, 0.6, 0.1],
         # [0.0, 0.1, 0.2, 0.7]]
    # B = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         # [0.5, 0.2, 0.1, 0.1, 0.05, 0.05],
         # [0.05, 0.05, 0.45, 0.05, 0.35, 0.05],
         # [0.25, 0.05, 0.05, 0.05, 0.05, 0.55]]
    # observ = [3, 2, 1, 2, 2, 6, 5]
    A = [[0.5, 0.3, 0.2],
         [0.0, 0.6, 0.4],
         [0.0, 0.0, 1.0]]
    B = [[0.7, 0.3],
         [0.4, 0.6],
         [0.8, 0.2]]
    observ = [1,2,1]
    pi = [0.9, 0.1, 0.0]
    theta = model(A,B,pi)
    for i in theta:
        print i
    print forward(theta, observ)
    print backward(theta, observ)

    print '-> Probability of the observed sequence is:'
    print likelihood(observ, theta)
    print '-> Infer the present: the probability of the present hidden state is:'
    print filtering(observ, theta)
    print '-> Infer the past: the probability of the past hidden state is:'
    for i in range(len(observ)-1):
        print i, smoothing(observ, theta, i)
    print '-> Most likely hidden path is: '
    path = viterbi(theta, observ)
    for i in path:
        print 'h%d->' % i,
    print '\b\b\b'

if __name__ == '__main__':
    main()