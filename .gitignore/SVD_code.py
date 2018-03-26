# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:02:32 2018

@author: Hansal_Trivedi
"""

from numpy import array
from numpy import diag
from numpy import zeros
from scipy.linalg import svd

A=array([
	[1,2,3,4,5,6,7,8,9,10],
	[11,12,13,14,15,16,17,18,19,20],
	[21,22,23,24,25,26,27,28,29,30]])

U,s,V = svd(A)

print(U)
print(s)
print(V)

Sigma = zeros((A.shape[0],A.shape[1]))

Sigma[:A.shape[0],:A.shape[0]]=diag(s)

n_select = 2

Sigma = Sigma[:,:n_select]

V=V[:n_select,:]

B = U.dot(Sigma.dot(V))
print(B)

T=U.dot(Sigma)
print(T)

T=A.dot(V.T)
print(T)
