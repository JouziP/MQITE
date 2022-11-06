#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:39:39 2022

@author: pejmanjouzdani
"""


import numpy as np
import pandas as pd


from qiskit import QuantumCircuit,  transpile
from qiskit import Aer

from Amplitude.AmplitudeFunctions.getAmplitudes import getAmplitudes
from Amplitude.AmplitudeFunctions.getIndexsFromExecute import getIndexsFromExecute

def computeAmplFromShots(circ, shots, j_ref, backend = 'qasm_simulator'):
    pass
    '''
    
    
    '''
   
    df_count = getIndexsFromExecute(circ, shots, backend)
    
    m1 = df_count['n_j'][j_ref]/shots #df_count[0].sum()
    
    return m1


if __name__=='__main__':
    '''
    Testing the function:
        create a circ_UQU circuit 
        
    
    '''
    
    from qiskit import QuantumCircuit
    
    from Amplitude.Amplitude import Amplitude
    from BasicFunctions.functions import getBinary, getState
    from MultiQubitGate.functions import multiqubit
    
    ################################################################
    ################################################################
    ##################### FOR TEST           #######################
    ################################################################
    ################################################################
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
    
    ################################################################
    ################################################################
    def getRandomQ(nspins):
        Q = np.random.randint(0,2, size=nspins)
        return Q
    
    ################################################################
    ################################################################
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
    
    ################################################################
    ################################################################
    
    seed = 1211
    np.random.seed(seed)
    
    ################################################################
    ################################################################
    nspins = 2    # >=2
    num_layers =2
    num_itr =1
    machine_precision = 10        
    shots = 1000000
    eta = 100
    significant_figures = 3#np.log10(np.sqrt(shots)).astype(int)
    
    circ_U = getRandomU(nspins, num_layers)    
    Q = getRandomQ(nspins)
    circ_UQU = getCircUQU(circ_U, Q)
    
    print(Q)
    
    ################################################################
    ################################################################
    ##    ### amplitude |c_j| computed from shots 
    localAmplitudeObj  =  Amplitude(circ_U, circ_UQU, shots, eta, significant_figures, machine_precision)
    
    df_ampl_bm, df_ampl, std_prob, drop_in_peak, m_support, m_support_rounded = localAmplitudeObj.computeAmplutudes()
    
    ######## EXCLUDE index 0
    try:
        df_ampl = df_ampl.drop(index=0)
        df_ampl_bm = df_ampl_bm.drop(index=0)
    except:
        pass
    
    print('df_ampl')
    print(df_ampl)
    print()
    print('df_ampl_bm')                
    print(df_ampl_bm)        
    print()
    
    
    ################################################################
    ################################################################
    ##    The steps involved in getImagPart
    
    j_ref = np.random.randint(0, 2**nspins)
    while j_ref in df_ampl.index:
        j_ref = np.random.randint(0, 2**nspins)
    
    print('j_ref = ' , j_ref)
    '''
    for the scenario where j_ref=0 is NOT in df_ampl.index
    or c_0 == 0
    '''
    
    
    circ_adj = QuantumCircuit(nspins+1)
    gamma =   np.pi/2
    circ_adj.ry(gamma, qubit=-1)  ### R_gamma
    circ_adj.x(qubit=-1)  ### X
    
    ### U
    ### attaches U to the q=1 ... q=n qubits, while q=0 is the ancillary 
    circ_adj = circ_adj.compose(QuantumCircuit.copy(circ_U) ) 
    
    ### control-Q ; Ancillary - n target 
    for (q,o) in enumerate(Q):
        if o==1:            
            circ_adj.cx(-1, q)
        if o==2:
            circ_adj.cy(-1, q)
        if o==3:
            circ_adj.cz(-1, q)
    
    ### U^    
    circ_adj = circ_adj.compose(QuantumCircuit.copy(circ_U).inverse())
    
    ### control-P_{0 j_ref}
    circ_adj.x(qubit=nspins)    
    J1 = list(getBinary(j_ref, nspins)) + [0]
    print(J1)
    for (q,o) in enumerate(J1):
        if o==1:            
            circ_adj.cx(nspins, q)                           
    circ_adj.x(nspins)  
    print(circ_adj)
    ### H on ancillary
    circ_adj.h(nspins)
    
    
    #################### for each j2  The T_{j_ref -> j2} is different 
    indexs = df_ampl.index  ### the observed bit strings from shots; j's
    parts_imag= [[ 0, 0]]  # ref
    part_indexs=[j_ref]
    
    
    for j2 in indexs[:10]:
        print('j2 = ', j2)
        #### T Gate
        # print('j2  : ', j2)  
        ####
        p_12_int = j2^j_ref                
        ## operator
        P_12 = getBinary(p_12_int, nspins).tolist()+[0] #bitstring array of p12
        mult_gate, op_count = multiqubit(P_12, np.pi/4) # turned into T gate
        circ_uhu_adj = circ_adj.compose( mult_gate ) #add to the circuit
        
        # print(circ_uhu_adj)
        ############################################
        #####  from shots
        m1_, df_count = computeAmplFromShots(circ_uhu_adj, shots, j_ref)
        m1_ = np.round(m1_, significant_figures)
        print('m1 from shots = ', m1_)
        ############################################
        
        
        ############################################
        #####  get components from the state vector
        # to be replaced by shots
        circ_state_adj  = getState(circ_uhu_adj, significant_figures)                                              
        # ancila == 0
        circ_state_adj = circ_state_adj.reshape(2, circ_state_adj.shape[0]//2).T[:, 0]        
        m1 = circ_state_adj[j_ref, 0]
        # print('sqrt(m1)   : ', m1)    
        m1 = (m1*m1.conjugate()).real.round(significant_figures)
        print('m1 from state vector = ', m1)
        ############################################
        
    
    
    
