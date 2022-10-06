#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:55:07 2022

@author: pej
"""




import numpy as np
from numpy import linalg as lg
from scipy.sparse.linalg import cg
import pandas as pd

X = np.matrix([[0.+0.j, 1.+0.j],
                       [1.+0.j, 0.+0.j]])
Y = np.matrix([[ 0.+0.j, -0.-1.j],
               [ 0.+1.j,  0.+0.j]])
Z = np.matrix([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]])
I2 = np.matrix([[1.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j]])


class  RandomH_k():
    '''
    assumes the H acts on q1 and q1+1
    '''
    
    def __init__(self, nspins, k):
        
        self.nspins = nspins
        self.k = k
        
   
    
    def getIden_k(self, q1, nspins, k):    
        assert(q1<nspins-k+1)
        assert(q1>=0)   
        IL=1
        IR=1
        for q in range(q1):
            if q==0:
                IL = I2
            else:
                IL = np.kron(IL,I2)
                
                
        for q in range(q1+k, nspins):
            if q==0:
                IR = I2
            else:
                IR = np.kron(IR,I2)
        return IR, IL
    
                
            
        
    def getRandomH(self, nspins, q1):
        
        k = self.k 
        
        # Hr = np.random.uniform(-1,1, size=[2**k, 2**k]).astype(complex)
        # Hi = np.random.uniform(-1,1, size=[2**k, 2**k]).astype(complex)        
        # H = Hr + 1j* Hi
    
        H = np.random.uniform(-1,1, size=[2**k, 2**k]).astype(complex)
        
        H = H.T.conjugate() + H
        # print(H.shape)
        IR, IL = self.getIden_k(q1, nspins, k)
        
        H = np.kron(np.kron(IL,  H), IR)        
        
        return H

class  RandomH_2():
    '''
    assumes the H acts on q1 and q1+1
    '''
    
    def __init__(self, nspins):
        
        self.nspins = nspins
        
        self.Os = [np.kron(X,I2), 
           np.kron(Y,I2),
           np.kron(Z,I2),
           np.kron(I2,X),
           np.kron(I2,Y),
           np.kron(I2,Z),
           np.kron(X,X),
           np.kron(X,Y),
           np.kron(X,Z),
           np.kron(Y,X),
           np.kron(Y,Y),
           np.kron(Y,Z),
           np.kron(Z,X),
           np.kron(Z,Y),
           np.kron(Z,Z)
          ]
        
   
    
    def getIden_2(self, q1, nspins):    
        assert(q1<nspins-1)
        assert(q1>=0)   
        IL=1
        IR=1
        for q in range(q1):
            if q==0:
                IL = I2
            else:
                IL = np.kron(IL,I2)
                
                
        for q in range(q1+2, nspins):
            if q==0:
                IR = I2
            else:
                IR = np.kron(IR,I2)
        return IR, IL
    
                
            
        
    def getRandomH(self, nspins, q1):
        
    
        params = np.random.uniform(-1,1, size=len(self.Os))
        for l in range(len(params)):
            if l==0:
                H =params[l] * self.Os[l]
            else:
                H+=params[l] * self.Os[l]
                
        IR, IL = self.getIden_2(q1, nspins)
        
        H = np.kron(np.kron(IL,  H), IR)        
        
        return H
        
        
        
    
    
if __name__=='__main__':
    np.random.seed(232)
    nspins=5
    q1 = 0
    k = 3
    
    
    r2 = RandomH_2(nspins)
    H = r2.getRandomH(nspins, q1)
    
    ##########
    # 
    rk = RandomH_k(nspins, k)
    H = rk.getRandomH(nspins, q1)
    
    
    
    
    
    
    
    