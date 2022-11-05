#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:25:23 2022

@author: pej
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:23:43 2022

@author: pej
"""


from scipy.linalg import schur
from time import time    , perf_counter, ctime
import os
import numpy as np
from numpy import linalg as lg
from scipy.sparse.linalg import cg
import pandas as pd
from matplotlib import pyplot as plt
from functools import wraps

import qiskit
from qiskit.extensions import IGate,XGate,YGate,ZGate
from qiskit import QuantumCircuit, execute, BasicAer, transpile
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import partial_trace, Statevector
from qiskit import Aer

from qiskit.providers.aer import QasmSimulator



from MultiQubitGate.functions import multiqubit

from BasicFunctions.test_functions import TestCirc

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r --->  %2.4f sec' %(f.__name__,   te-ts) )
        return result
    return wrap

X_op = np.matrix([[0.+0.j, 1.+0.j],
                       [1.+0.j, 0.+0.j]])
Y_op = np.matrix([[ 0.+0.j, -0.-1.j],
               [ 0.+1.j,  0.+0.j]])
Z_op = np.matrix([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]])
I2 = np.matrix([[1.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j]])




###############################################################################
@TestCirc
def getQCirc(circ, Q):    
    #############
    # Qiskit error
    if not isinstance(circ, QuantumCircuit):
        raise qiskit.circuit.exceptions.CircuitError('The circuit is not an instance of the QuantumCircuit')
    
    # TypeError
    if not isinstance(Q, np.ndarray):
        if not isinstance(Q, list):
            raise TypeError(' the provided Pauli string operator is not am instance of list or numpy array ')
    #############

    
    circ_Q = QuantumCircuit.copy(circ)

    #############
    # ValueError
    ### the number of qubits must be the same as number of operators in Q
    if not circ_Q.num_qubits == len(Q):
        raise ValueError('the number of qubits in the circuit is not the same as the  number of Pauli operators in Q')
    #############
    
    
    for (q, o) in enumerate(Q):
        if o==0:
            pass
        if o==1:
            circ_Q.x(q)
        if o==2:
            circ_Q.y(q)
        if o==3:
            circ_Q.z(q)
            
    return circ_Q

@TestCirc
def getUQUCirc(circ, circ_Q):
    circ_UQU = QuantumCircuit.copy(circ_Q)  ## QU|0>
    circ_UQU = circ_UQU.compose(QuantumCircuit.copy(circ.inverse()) )## U^QU|0>
    return circ_UQU ## U^QU|0>

###############################################################################













def getExact(hs, ws):
    ######### exact
    #################
    hs_mtx = []
    for (i,h) in enumerate(hs):
        h_mtx = geth_mtx(h)
        hs_mtx.append(h_mtx)
    
    for (i,h_mtx) in enumerate(hs_mtx):
        if i==0:
            H_mtx = ws[i]*h_mtx
        else:
            H_mtx += ws[i]*h_mtx
    e,v = lg.eigh(H_mtx)
    
    
    E_exact = e[0]
    
    print('E exact = ' , E_exact)
    
    return hs_mtx, H_mtx, E_exact


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
        ###############
        if i==0:
            h_mtx = O 
        else:
            h_mtx = np.kron(O, h_mtx)
    return h_mtx

##############################
# @timing
def getITE(circ_h, circ, delta, machine_precision):
    psi_t = getState(circ, machine_precision)
    h_psi_t = getState(circ_h, machine_precision)
    
    psi_t_delta = psi_t - delta * h_psi_t    
    norm2 = psi_t_delta.T.conjugate().dot(psi_t_delta)[0,0].real
    norm = np.sqrt(norm2)
    
    psi_t_delta = normalize(psi_t_delta)
    
    return psi_t_delta, norm



def getOverlap(a, b):
    f = a.T.conjugate().dot(b)[0,0]
    return f

def getExpec(a, H):
    
    val = a.T.conjugate().dot(H).dot(a)
    return val[0,0].real


# @timing
def getState(circ, machine_precision):
    return np.matrix(Statevector(circ)).reshape(2**circ.num_qubits,1).round(machine_precision)

def normalize(psi):
    norm2 = psi.T.conjugate().dot(psi)[0,0].real
    psi = psi/np.sqrt(norm2)
    return psi

########################################


def getBinary(j, nspins):
    string = np.array(list(np.binary_repr(j, nspins))).astype(int)[::-1]
    return string





##################################
def getGateStats(multiqubit_stats, Q, iQ):
    # print('Q = ' , Q)
    # print('js = ' , multiqubit_stats.index)    
    # print('num_j = ' , multiqubit_stats.shape[0])    
    ##### stats    
    # print() 
    # print('multiqubit_stats ')    
    
    stats_Q = pd.DataFrame(multiqubit_stats.sum()).T
    stats_Q.index=[iQ]
    
    stats_Q['total'] = [stats_Q.sum(axis=1).values[0]]
    
    # print(stats_Q)    
    
    # #### 
    # print() 
    # ####
    # gate_stats_new = circ_new.decompose().count_ops()    
    # df_gates_new =  pd.DataFrame([gate_stats_new])
    # print('After update = ')        
    # print(df_gates_new)    
    # print() 
    # change =  df_gates_new- df_gates_old
    # print('change ')  
    # print(change)  
    # print('-------- ') 
    # print() 
    # print() 
    # ####
    return stats_Q


#################
def getNum_j_UHU(psi_t, H_mtx, U_t, significant_digits):
    
    psi_sim_UHU_t = (U_t.T.conjugate().dot(H_mtx)).dot(psi_t)
    psi_sim_UHU_t = np.array(psi_sim_UHU_t.T.tolist()[0])
    prob_j_t = np.array(list(map(lambda t: (t.conjugate() * t ).real   ,
                   psi_sim_UHU_t)))            
    prob_j_t = prob_j_t.round(significant_digits)                                    
    prob_j_t = prob_j_t/sum(prob_j_t)            
    prob_j_t = prob_j_t.round(significant_digits)                        
    num_j_UHU_t = len(prob_j_t.nonzero()[0])

    return num_j_UHU_t


def getETHInfo(U_t, H_mtx,  delta, time):
    ### out 
    E_eth = 0 
    
    ######
    nspins = int(np.log2(H_mtx.shape[0]))
    circ0 = QuantumCircuit(nspins)
    psi_init = getState(circ0, 5)
    
    T, Z= schur(U_t)
    
    es_z = ((1j * np.log(np.diag(T))).real).round(5)
    es_z_sorted = np.sort(es_z)
    dE = 1./(time + delta)
    vals_in_interval = es_z_sorted[es_z_sorted< es_z_sorted[-1]+dE]
    indxs = [np.where(es_z==vals_in_interval[i])[0][0]\
                  for i in range(vals_in_interval.shape[0])]        
    Z = Z[:, indxs]
    
    H_zz = np.diag( Z.T.conjugate().dot(H_mtx) .dot(Z) )
    c_zs = (psi_init.T.conjugate().dot(Z)).tolist()[0]
    
    E_eth = sum(np.array(list(map(lambda x,y: (x*x.conjugate()).real * y.real , c_zs, H_zz))))
    
    
    val = psi_init.T.conjugate().dot((U_t.conjugate().T.dot(H_mtx).dot(U_t))).dot(psi_init)
    val = val[0,0].real
    print('E_Q_eth=', E_eth.round(5))
    print('E_Q= ', val.round(5))
    
    
    return E_eth




if __name__=='__main__':
    
    number_qubits= 3
    
    Q = [1,2,3]
    
    circ = QuantumCircuit(number_qubits)
    
    print(getQCirc(circ, Q))