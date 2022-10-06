#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:34:50 2022

@author: pejmanjouzdani
"""


      
from time import time    
import os
import numpy as np
from numpy import linalg as lg
from scipy.sparse.linalg import cg
import pandas as pd
from matplotlib import pyplot as plt
from functools import wraps



from qiskit.extensions import IGate,XGate,YGate,ZGate
from qiskit import QuantumCircuit, execute, BasicAer
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import partial_trace, Statevector

I2  = IGate().to_matrix()
X = XGate().to_matrix()
Z = ZGate().to_matrix()




X_op = np.matrix([[0.+0.j, 1.+0.j],
                       [1.+0.j, 0.+0.j]])
Y_op = np.matrix([[ 0.+0.j, -0.-1.j],
               [ 0.+1.j,  0.+0.j]])
Z_op = np.matrix([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]])
I2 = np.matrix([[1.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j]])



def geth_mtx(h):
    for (i,o) in enumerate(h):
        if o == 0:
            O = I2
        if o==1:
            O = X_op
        if o==2:
            O = Y_op
        if o==3:
            O = Z_op
        if i==0:
            h_mtx = O 
        else:
            h_mtx = np.kron(O, h_mtx)
    return h_mtx



R2 = UnitaryGate((np.kron(I2, I2) - 1j * np.kron(Z, X)) * 1./(np.sqrt(2)) ) 
R1 = UnitaryGate((I2 + 1j * Z) * 1./(np.sqrt(2)) )

R2dg = UnitaryGate(  ((np.kron(I2, I2) + 1j * np.kron(Z, X)) * 1./(np.sqrt(2)) ))
R1dg = UnitaryGate(   ((I2 - 1j * Z) * 1./(np.sqrt(2)) ))



Uxx = lambda theta :  (np.kron(I2, I2) * np.cos(theta) + 1j *np.sin(theta) * np.kron(X, X))


 
def findPairs(h):
    pairs =[]
    nspins = len(h)
    
    
    q1=0
    while q1<nspins:
        if h[q1]==0:
            q1+=1 
        else:
            break
        
    if q1==nspins-1:
        return pairs
    
    # print(q1)
    
    
    q2 = q1+1        
    while q2<nspins :
        if h[q2]==0:
            q2+=1                
        else:
            pairs.append([q1,q2])
            q1=q2
            q2=q1+1
            
    
    return pairs

def multiqubit(h, theta):
    nspins = len(h)
    circ = QuantumCircuit(nspins)
    num_opr = len(np.nonzero(h)[0])
    
    if num_opr >1:
        ######### U(1) --> X
        for q in range(nspins):
            if h[q]==0:
                pass
            if h[q]==1:
                pass
            if h[q]==2:            
                circ.h(q)
                circ.s(q)
            if h[q]==3:
                circ.h(q)
        
        
        ##### (I-iZ) (I-i XZ)
        pairs = findPairs(h)
        for pair in pairs[:-1]:
            circ.unitary(R2dg, pair )
            circ.unitary(R1dg, pair[1])
            
        circ.unitary(UnitaryGate(Uxx(theta)), pairs[-1])
        
        ##### (I-iZ) (I-i XZ)
        pairs = findPairs(h)
        for pair in pairs[:-1][::-1]:        
            circ.unitary(R1, pair[1])
            circ.unitary(R2, pair )
        
         ######### U(1) --> X
        for q in range(nspins):
            if h[q]==0:
                pass
            if h[q]==1:
                pass
            if h[q]==2:  
                circ.sdg(q)
                circ.h(q)
            if h[q]==3:
                circ.h(q)
    else:
        for q in range(nspins):
            if h[q]==0:
                pass
            if h[q]==1:
                circ.rx(-theta * 2, q)    
            if h[q]==2:            
                circ.ry(-theta * 2, q)
            
    
    ops_count = circ.decompose().count_ops()
    
    return circ, ops_count

if __name__ == '__main__':
    from qiskit import Aer
    backend = Aer.get_backend('unitary_simulator')
    h = [1,1,0, 0, 2] 
    theta = np.random.uniform(-1,1)
    print(theta)
    nspins = len(h)       
    h_mtx = geth_mtx(h)
    
    IN = np.identity(2**nspins, dtype=complex)
    
    U_h = IN * np.cos(theta)  + 1j * np.sin(theta) * h_mtx
    U_h = U_h.round(4)
    
        
    circ_h_replica, ops_count = multiqubit(h, theta)
    
    
    print(circ_h_replica.draw())
    job = execute(circ_h_replica, backend=backend )
    result = job.result()
    U_h_replica = np.matrix(result.get_unitary( circ_h_replica)).round(4)
    
    print(np.sum(U_h==U_h_replica) == 4**nspins )
        
    
    
    
            
            