#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:20:16 2022

@author: pej
"""

import pandas as pd
import numpy as np

from Phase.getCosPhasesNoC0_test import getCosPhasesNoC0
from Phase.getSinPhasesNoC0_test import getSinPhasesNoC0

def computePhaseStateVec(df_ampl,  
                         df_comp_bm, 
                         circ,
                         Q,
                         circ_uhu, 
                         significant_figures, 
                         nspins):
    '''
    
    
    
    Parameters
    ----------
    df_ampl : Panda DataFrame
        it contains one column, with label 1, which are the amplitude of 
        the observe indexs. the indexs are the js, while the column 
        values are |c_j|
        
    df_comp_bm : Panda DataFrame
        it's a benchmark of the cj's
    
    circ : Qiskit circuit instruction
        DESCRIPTION.
        
    Q : list
        list out of  {0,1,2,3}. It represents a Pauli string; Q=[0,1,1]<=>IXX
        
    circ_uhu : Qiskit circuit instruction
        DESCRIPTION.
        
    significant_figures : int
        DESCRIPTION.
    nspins : int
        num. spins / qubits

    Returns
    -------
    df_cj: panda dataframe
        the c_j simulated using the sub-algorithm in Appendix B

    '''
    
   
    #### choose the ref 
    j_ref = np.random.randint(0, 2**nspins)
    while j_ref in df_ampl.index:
        j_ref = np.random.randint(0, 2**nspins)
   
       
    sin_phases=getSinPhasesNoC0(df_ampl,  df_comp_bm, 
                                circ, Q,
                                significant_figures, nspins, j_ref)
    cos_phases=getCosPhasesNoC0(df_ampl,  df_comp_bm, 
                                circ, Q,
                                significant_figures, nspins, j_ref)

    df_cj = pd.DataFrame(cos_phases['c_real_sim'] + 1j* sin_phases['c_imag_sim'])
    df_cj.columns=[0]
              
   
    return df_cj







if __name__=='__main__':
    '''
    Try to catch scenarios where there are significant errors
    
    '''
    
    pass


      
    import numpy as np
    import pandas as pd
    
    
    
    from qiskit import QuantumCircuit
    from MultiQubitGate.functions import multiqubit
    
    
    from BasicFunctions.functions import getBinary, getState
        
    nspins = 10
    n_h = nspins
    ngates = nspins
    num_layers = 3
    

    
    machine_precision = 10            
    shots = 1000000
    eta =100
    significant_figures = 3
    
    seed = 365
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
    # print(circ)
    ####### random Q
    Q = getRandomQ(nspins)   
    print(Q)
    #### UQU
    circ_uhu = getCircUQU(circ, Q)
    
    from Amplitude.computeAmplitudeV1 import computeAmplitude
    
    
    df_comp_bm, df_ampl, m_support, std_prob, drop_in_peak, df_ampl_org = computeAmplitude(circ_uhu, 
                                                                                           shots,
                                                                                           eta, 
                                                                                           significant_figures, 
                                                                                           machine_precision)
    print('df_ampl')
    print(df_ampl)
    print()
    print('df_ampl_bm')        
    print(pd.DataFrame(df_comp_bm[0].apply(lambda x: np.sqrt(  (x*x.conjugate()).real ) )))        
    print()
    
    significant_digits = int((1/2) * np.log10(shots/(m_support)))+2
    
    
    
    df_cj = computePhaseStateVec(df_ampl,  
                          df_comp_bm, 
                          circ,
                          Q,
                          circ_uhu, 
                          significant_digits, 
                          nspins)
    
    print('df_comp_bm  --  df_cj_sim')
    df = pd.concat((df_comp_bm, df_cj), axis=1)
    df.columns=['exact', 'sim']
    print(df)
    print()
    
    dff = (df_comp_bm-df_cj)
    
    print('dff rounded to significant_figures')
    print(dff[0].round(significant_figures))
    print( '\n\n#### L2 norm error/ num. componenets')
    print( (dff[0].apply(lambda x: np.sqrt( (np.round(x,significant_figures)*np.round(x,significant_figures).conjugate()).real ) ).sum()) / df_comp_bm.shape[0] )








