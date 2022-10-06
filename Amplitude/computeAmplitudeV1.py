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
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit,  transpile
from qiskit.quantum_info import  Statevector
from qiskit.providers.aer import QasmSimulator, AerSimulator
from qiskit import Aer


from BasicFunctions.functions import getState, timing

        
# @timing
def getIndexsFromExecute(circ, shots):
    '''
    appends classical wires to record measurements
    execute the circuit 'shots' time
    returns the observed bit-strings and distributions (counts)
    
    [AerSimulator('aer_simulator'),
     AerSimulator('aer_simulator_statevector'),
     AerSimulator('aer_simulator_density_matrix'),
     AerSimulator('aer_simulator_stabilizer'),
     AerSimulator('aer_simulator_matrix_product_state'),
     AerSimulator('aer_simulator_extended_stabilizer'),
     AerSimulator('aer_simulator_unitary'),
     AerSimulator('aer_simulator_superop'),
     QasmSimulator('qasm_simulator'),
     StatevectorSimulator('statevector_simulator'),
     UnitarySimulator('unitary_simulator'),
     PulseSimulator('pulse_simulator')]
    
    
    '''
   
    simulator = Aer.get_backend('qasm_simulator')
   
    
    ####################
    
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
# def computeAmplitude(circ_uhu, shots, significant_digits_add):
def computeAmplitude(circ_uhu, shots, eta, significant_figures, machine_precision=10):
    '''
    

    Parameters
    ----------
    circ_uhu : Qiskit circuit of U^ Q U 
        
    shots : Int
        number of shots
    
    eta : int
        number of dominant components/amplitudes
    
    significant_figures : int
        significant figure == \epsilon in the precision 10^-\epsilon (approx)
    
    machine_precision : int
        significant figure in the benchmark values

    Returns
    -------
    df_comp_bm : panda dataframe
        is the benchmark value (ideal) of the components from state-vector
        rounded to the precision 
        
    df_ampl : panda dataframe
        the amplitude obtained from the shots .
        
    m_support : int
        number of components that are non-zero after executions of the circuit
        
    std_prob : float
        standard deviation in the amplitude 
    
    drop_in_peak : float
        the difference between the largest and smallest observed amplitude
        
    df_ampl_org : TYPE
        DESCRIPTION.
        
   
    '''
    #### copy for bm; ensure there is no mixing
    circ = QuantumCircuit.copy(circ_uhu) 
    
    ############################################################# Simulation 
    
    #################### |cj| and js upto significant digits
    counts, j_indxs = getIndexsFromExecute(circ_uhu, shots)        
    df_count = pd.DataFrame([counts.int_outcomes()]).T    
        
    ### metrics    
    m_support = df_count.shape[0]    
    print('m_support = ', m_support)
        
    ### amplitudes |cj|
    df_ampl = (df_count/df_count.sum()).apply(lambda x: np.sqrt(x))  
    df_ampl = df_ampl.sort_values(0, ascending=False)
    df_ampl_org = pd.DataFrame.copy(df_ampl) # the amplitudes before rounding
    
    #################### some other metrics 
    m_support_rounded = pd.DataFrame.sum(df_ampl_org.round(significant_figures)!=0)[0]
    print('m_support_rounded = ', m_support_rounded)
    
    ###  the observable (metric) standard deviation 
    std_prob = df_ampl.std()[0]    
    if pd.isna(std_prob):
        std_prob = 0     
    print('std_prob = ', std_prob)
    
    ###  the observable (metric) \Delta^*
    try:
        drop_in_peak = df_ampl_org.values[0,0] - df_ampl_org.values[1,0]
    except:
        drop_in_peak = df_ampl_org.values[0,0]
    
    ####################  pick the eta dominant amplitudes |cj|
    df_ampl = df_ampl[:eta]
    
    #####  making sure they are all non-zero
    df_ampl = df_ampl.loc[df_ampl[0]!=0]
    
    #### js of non-zero elements 
    j_list = df_ampl.index.tolist()
    
    #################### Benchmark 
    circ_state = getState(circ, machine_precision)   
    df_comp_bm =  pd.DataFrame.copy(df_ampl)    
    df_comp_bm.loc[df_comp_bm.index]=circ_state[j_list, :].round(significant_figures).T.tolist()[0]        
    ####################
    
    ### sort    
    df_comp_bm = df_comp_bm.sort_index()
    df_ampl = df_ampl.sort_index()
    
    ############
    return df_comp_bm, df_ampl, m_support, std_prob, drop_in_peak, df_ampl_org







if __name__=='__main__':
    pass


    from matplotlib import pyplot as plt
    nspins = 12
    
    num_layers =2
    num_itr =1
    machine_precision = 10        

    shots = 1000
    eta = 100
    significant_figures = 3#np.log10(np.sqrt(shots)).astype(int)
    
    
    #####
    seed = 1253
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
    m_support_arr= []    
    for r in range(num_itr):
        ######## random circ
        circ = getRandomU(nspins, num_layers)
        print(circ)
        ####### random Q
        Q = getRandomQ(nspins)    
        #### UQU
        circ_uhu = getCircUQU(circ, Q)
        df_comp, df_ampl, m_support, std_prob,\
        drop_in_peak, df_ampl_org = computeAmplitude(circ_uhu, shots, eta, significant_figures )
        print()
        print('df_ampl_bm')        
        print(pd.DataFrame(df_comp[0].apply(lambda x: np.sqrt(  (x*x.conjugate()).real ) )))        
        print()
        print('df_ampl')        
        print(df_ampl)
        print()
        print('df_ampl - df_ampl_bm')        
        dff = pd.DataFrame(df_comp[0].apply(lambda x: np.sqrt(  (x*x.conjugate()).real ) ))-df_ampl
        print( dff[0].apply(lambda x: np.sqrt(x**2) ).sum() )
        
        
        norm = sum([(df_comp[0][i]*df_comp[0][i].conjugate()).real for i in df_comp.index])
        # print(norm)
        
        # print(m_support/2**nspins)
        m_support_arr.append(m_support/2**nspins)
    
    m_support_arr = np.array(m_support_arr)
    # plt.hist(m_support_arr, bins=num_itr)
    