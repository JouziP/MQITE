#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:12:01 2022

@author: pej
"""




from functools import wraps      
from time import time    , perf_counter, ctime
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit,  transpile
from qiskit.quantum_info import  Statevector
from qiskit.providers.aer import QasmSimulator



from BasicFunctions.functions import getState, timing

        
# @timing
def getIndexsFromExecute(circ, shots):
    '''
    appends classical wires to record measurements
    execute the circuit 'shots' time
    returns the observed bit-strings and distributions (counts)
    '''
    
    simulator = QasmSimulator(method='statevector')
    
    num_qubits = circ.num_qubits
    circ_meas = QuantumCircuit(num_qubits, num_qubits)
    circ_meas = circ_meas.compose(circ)
    
    circ_meas.measure(
        [q for q in range(num_qubits)],
        [q for q in range(num_qubits)]
        )
    
    compiled_circuit = transpile(circ_meas, simulator)
        
    job = simulator.run(compiled_circuit, shots=shots)
    
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    
    j_indxs = np.sort(list([key for key in counts.int_outcomes().keys()]))
            
    return [counts, j_indxs]
#################################################################################
def computeAmplitude(circ_uhu, shots, significant_digits_add):
    
    machine_precision = 10
    circ = QuantumCircuit.copy(circ_uhu) 
    circ_state = getState(circ, machine_precision)   
    
    ### cj and js upto significant digits
    counts, j_indxs = getIndexsFromExecute(circ_uhu, shots)        
    df_count = pd.DataFrame([counts.int_outcomes()]).T    
    ########################
    # sig_figure_init = np.log10(np.sqrt(shots)).astype(int)
    m_support = df_count.shape[0]
    # print('m_support = ', m_support)
    sig_figure = int((1/2) * np.log10(shots/(m_support)))+significant_digits_add
    # print()
    # print('sig_figure, sig_figure_init')
    # print(sig_figure, sig_figure_init)
    
    
    #############################################
    ### amplitudes |cj|
    df_ampl = (df_count/df_count.sum()).apply(lambda x: np.sqrt(x))  
    df_ampl = df_ampl.sort_values(0, ascending=False)
    
    # print(df_ampl.shape[0])
    df_ampl = df_ampl.round(   sig_figure  )
    df_ampl = df_ampl.loc[df_ampl[0]!=0]
    # df_ampl = df_ampl[:num_states_in_support]
    # print(df_ampl.shape[0])
    
    
    #### js of non-zero elements after rounding to sig.digit.-1        
    j_list = df_ampl.index.tolist()
    # m_support_fin = len(j_list)
    
    ###### eventually uses the state vector data
    df_comp =  pd.DataFrame.copy(df_ampl)
    df_comp.loc[df_comp.index]=circ_state[j_list, :].round(sig_figure).T.tolist()[0]    
    # print(df_comp)
    
    
    #############################################
    # ##### for bm
    # prob_dist = [  (ci[0,0]*ci[0,0].conjugate()).real for ci in circ_state]
    # prob_dist = pd.DataFrame(prob_dist)
    # # print(prob_dist)
    # prob_dist = prob_dist.round(sig_figure)
    # j_list_expect = prob_dist.loc[prob_dist[0]!=0].index   
    # prob_dist = prob_dist.loc[j_list_expect]
    # # print(prob_dist)
    # df_comp =  pd.DataFrame.copy(prob_dist)
    # df_comp.loc[df_comp.index]=circ_state[prob_dist.index, :].round(sig_figure  ).T.tolist()[0] 
    # # df_comp = df_comp.sort_index()
    
    # df_ampl =  pd.DataFrame.copy(prob_dist)
    
    ################################################
    ### sort    
    df_comp = df_comp.sort_index()
    df_ampl = df_ampl.sort_index()
    
    ############
    return df_comp, df_ampl, m_support







if __name__=='__main__':
    pass


    
    nspins = 5
    n_h = nspins
    ngates = nspins
    num_layers = 2
    num_itr =3
    W_max =1
    W_min = -1
    Hz_max=1
    Hz_min=-1
    k = 2
    delta0 = 0.1
    T = num_itr * delta0
    machine_precision = 10        
    chi  = 1
    shots = 10000
    significant_digits = 6#np.log10(np.sqrt(shots)).astype(int)
    delta = delta0
    seed = 2212
    ######
    np.random.seed(seed)
    
    
    ################################################################
    ################################################################
    ##################### FOR TEST           #######################
    ################################################################
    ################################################################
    ################################################################

    def getRandomU(nspins, num_layers=10):
        circ = QuantumCircuit(nspins)
        for l in range(num_layers):
            for i in range(nspins):
                ##############
                q=np.random.randint(nspins)
                g=np.random.randint(1, 4)
                p=np.random.uniform(-1,1)
                if g==1:
                    circ.rx(p,q)
                if g==2:
                    circ.ry(p,q)
                if g==2:
                    circ.rz(p,q)
                ##############
                q=np.random.randint(nspins-1)
                circ.cnot(q, q+1)
        return circ            
    
    def getRandomQ(nspins):
        Q = np.random.randint(0,2, size=nspins)
        return Q
    def getCircUQU(circ, Q):
        circ_uhu=QuantumCircuit.copy(circ)
        for (q,o) in enumerate(Q):
            if o==0:
                pass
            if o==1:
                circ_uhu.x(q)
            if o==2:
                circ_uhu.y(q)
            if o==3:
                circ_uhu.z(q)     
                
        
        circ_uhu=circ_uhu.compose(QuantumCircuit.copy(circ).inverse())
        return circ_uhu
    
    ################################# TEST
    ######## random circ
    circ = getRandomU(nspins, num_layers)
    print(circ)
    ####### random Q
    Q = getRandomQ(nspins)    
    #### UQU
    circ_uhu = getCircUQU(circ, Q)
    df_comp, df_ampl, m_support= computeAmplitude(circ_uhu, significant_digits, shots)
    print(df_comp)
    print(df_ampl)