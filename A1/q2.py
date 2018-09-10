# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses
 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           In our dataset, d = 14
           The first data point x[0,:]
           The first feature(intercept x[:,0]
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    # TODO
    A_i = np.divide((l2(test_datum.T,x_train)),((-2)*(tau**2)))
    A = np.diag(np.exp(np.subtract(A_i,logsumexp(A_i)))[0]).T
    #Use np.linalg.solve to calculate w
    #left part (XTAX+lam*I)
    left = np.add(np.dot(np.dot(x_train.T,A), x_train), 
                  np.dot(np.identity(x_train.shape[1]),lam))
    #right part (XTAy)
    right = np.dot(np.dot(x_train.T,A), y_train)
    w = np.linalg.solve(left,right)
    return np.dot(test_datum.T, w)


def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''    
    num_of_obs = np.round((x.shape[0])/k)
    loss = np.zeros([k,len(taus)])
    for i in range(k):
        start = i*num_of_obs 
        end = (i+1)*num_of_obs
        test_index = idx[start:end]
        training_index = np.setdiff1d(idx,test_index)
        X_training = x[training_index]
        X_test = x[test_index]
        y_training = y[training_index]
        y_test = y[test_index]
        loss[i,:] = run_on_fold(X_test, y_test, X_training, y_training, taus)
    return np.mean(loss, axis=0)


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.plot(taus, losses, ".")
    plt.xlabel("tau")
    plt.ylabel("Average Loss")    
    print("min loss = {}".format(losses.min()))

