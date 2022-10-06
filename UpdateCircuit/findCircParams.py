#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 19:47:39 2022

@author: pejmanjouzdani
"""



import numpy as np
import pandas as pd


from Amplitude.Amplitude import Amplitude


class FindCircParams:
    def __init__(self, circ,
                          iQ, 
                          Q,
                          circ_UQU,                 
                          delta_w, 
                          eta,                    
                          shots, 
                          nspins,                                                                       
                          significant_figures, 
                          machine_precision=10):
        self.circ = circ
        self.iQ = iQ
        self.Q = Q
        self.circ_UQU = circ_UQU
        self.delta_w = delta_w
        self.eta = eta
        self.shots = shots
        self.nspins = nspins
        self.significant_figures = significant_figures
        self.machine_precision = machine_precision
        
        localAmplitudeObj  =  Amplitude(circ, circ_UQU, shots, eta, significant_figures, machine_precision)
        
        
    def computeParameters(self ):
        pass
        
        df_ampl_bm, df_ampl, std_prob, drop_in_peak, m_support, m_support_rounded = self.localAmplitudeObj.computeAmplutudes()
        
        ### if nothing observed 
        if df_ampl.empty==True:
            return [[], [], m_support]
        
        # ### if observed 
        # else:
        #     ### if nothing observed 
        #     df_cj = computePhaseStateVec(  df_ampl,  
        #                                   df_comp_bm, 
        #                                   circ,
        #                                   Q,
        #                                   circ_UQU, 
        #                                   significant_figures, 
        #                                   nspins)            
    
        #     ### if at least one y_j is non-zero
        #     try:                        
        #         try:
        #             Q_expect = df_cj.loc[0][0].real
        #         except:
        #             Q_expect = 0 
                
        #         ### norm    
        #         norm = 1- 2*delta_w * Q_expect + delta_w ** 2
                
        #         ### y_j
        #         df_cj= df_cj.loc[df_ampl.index]            
        #         df_y_j =  ((df_cj)[0] * -(delta_w/norm) )                        
        #         df_y_j = pd.DataFrame(df_y_j, columns=[0])
                
        #         ### j_list and y_list
        #         j_list = df_y_j.loc[df_y_j[0]!=0].index
        #         df_y_j = df_y_j.loc[j_list]            
        #         # remove 0
        #         try:
        #             df_y_j = df_y_j.drop(index=0)
        #         except:
        #             pass            
        #         list_y_j = df_y_j[0].tolist() 
        #         list_j = df_y_j.index.tolist()
        #     ### if all y_j's are zero
        #     except:
        #         return [[], [], m_support]
    
        #     ### return successfully
        #     return [list_j, list_y_j, m_support, std_prob, drop_in_peak,  df_ampl_org]
        

if __name__=='__main__':
    pass

    from qiskit import QuantumCircuit
    
   
    
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
    
    seed = 1253
    np.random.seed(seed)
    
    ################################################################
    ################################################################
    nspins = 12    
    num_layers =2
    num_itr =1
    machine_precision = 10        
    shots = 1000
    eta = 100
    significant_figures = 3#np.log10(np.sqrt(shots)).astype(int)
    
    circ = getRandomU(nspins, num_layers)
    Q = getRandomQ(nspins)
    circ_uhu = getCircUQU(circ, Q)
            