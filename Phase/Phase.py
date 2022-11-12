#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:40:38 2022

@author: pejmanjouzdani
"""

# # from SimulateMQITE.log_config import logger

import pandas as pd
import numpy as np
from qiskit import QuantumCircuit



from MultiQubitGate.functions import multiqubit
from BasicFunctions.functions import getBinary

from Phase.PhaseFunctions.computeAmplFromShots import computeAmplFromShots


class Phase:
    ##### CONSTRUCTOR
    def __init__(self, 
                  amplObj,
                  nspins, 
                  circ, 
                  Q,
                  significant_figure, 
                  shots,
                  machine_precision=10,
                  j_ref=None,
                  gamma= np.pi/10):
        
        #### static
        self.amplObj            = amplObj
        self.nspins             = nspins
        self.circ               = circ ## the U_t circuit
        self.Q                  = Q # the Q Pauli string given in form of an array of 0,...,3
        self.significant_figure = significant_figure # significant_figures
        self.shots              = shots # 
        self.machine_precision  = machine_precision        
        self.df_ampl            = amplObj.df_ampl# |c_j| in U^QU|0>=\sum_j c_j|j>
        self.gamma= gamma
        
        if j_ref==None:
            #### choose the ref 
            j_ref = np.random.randint(0, 2**nspins)
            while j_ref in self.df_ampl.index:
                j_ref = np.random.randint(0, 2**nspins)            
        else:
            pass
        self.j_ref = j_ref
    

    
    ##### OVERWRITING () OPERATOR     
    def __call__(self, TestLevel=0):
        
        self.getImagPart()
        self.getRealPart()
        
            
            
    
    
    
    
    ##### TURN C_J TO Y_J
    def getY(self, delta):
        '''
        uses the c_j^(r) and imag to compute the y_j_r and imaginary
        '''
        NotImplemented
        
        try:
            ## <0| U^ Q U |0>
            Q_expect = self.df_ampl.loc[0][0].real
        except:
            Q_expect = 0 
        
        ### norm    
        norm = 1- 2*delta * Q_expect + delta ** 2
        
        ### y_j
        df_y_j =  ((self.df_ampl)[0] * -(delta/norm) )                        
        df_y_j =  pd.DataFrame(df_y_j, columns=[0])
        
        ### j_list and y_list
        j_list = df_y_j.loc[df_y_j[0]!=0].index
        df_y_j = df_y_j.loc[j_list]            
        # remove 0 double check
        try:
            df_y_j = df_y_j.drop(index=0)
        except:
            pass            
        self.ys_j = df_y_j[0].tolist() 
        self.js = df_y_j.index.tolist()
        
        return [self.js, self.ys_j]
        
    
    
    ##### Q. COMPUTE  IMAG PART OF C_J
    def getImagPart(self,):
        
        circ_adj = self.getImagPart_base_circ(self.nspins, self.circ, self.Q, self.j_ref, self.gamma)
        
        #################### for each j2  The T_{j_ref -> j2} is different 
        indexs = self.df_ampl.index  ### the observed bit strings from shots; j's
        parts_imag= [] #[[ 0 ]]  # ref has c_j_ref = 0 + 0j
        part_indexs=[] #[self.j_ref]
        m1s = []
        c2s = []
        
        for j2 in indexs:
            circ_uhu_adj = self.getImagPart_ref_circ(self.j_ref, j2, self.nspins,  circ_adj)
                        
            #####  amplitude of m_ref from shots 
            m_ref = computeAmplFromShots(circ_uhu_adj, self.shots, self.j_ref)
            ### collect results            
            m1s.append(m_ref)
            ###
            m_ref = np.round(m_ref, self.significant_figure)
            
            
            #### amplitude  from shots            
            c2 = self.df_ampl['|c_j|'][j2]
            ### collect results
            c2s.append(c2)
            ####
            c2_2 = c2**2 ### |c_j|^2
            self.c2_2    = np.round(c2_2, self.significant_figure)
            
            #### compute the sin of theta        
            imag_part = (m_ref - (1/4) * self.c2_2  *\
                          (np.cos(self.gamma/2)**2) - (1/4)*(np.sin(self.gamma/2))**2 )/\
                            ((-1/2) * np.cos(self.gamma/2) * np.sin(self.gamma/2)) 
        
            #### round to allowed prcision
            imag_part = np.round(imag_part, self.significant_figure)
            
            ### collect results
            parts_imag.append([ imag_part ])
            part_indexs.append(j2)
        
        ### final results
        parts_imag = pd.DataFrame(parts_imag, index= part_indexs).round(self.significant_figure)
        parts_imag.columns=[ 'c_imag_sim' ]    
    
        ###
        m1s = pd.DataFrame(m1s, index= part_indexs)
        m1s.columns=[ 'm1s-imag-sim' ]  
        c2s = pd.DataFrame(c2s, index= part_indexs)       
        c2s.columns=[ 'c2s-imag-sim' ]  
        
        ### attribute to the obj
        self.parts_imag = parts_imag
        self.parts_real_m1s =  m1s
        self.parts_real_c2s =  c2s
        
    
    
    ##### Q. COMPUTE  REAL PART OF C_J
    def getRealPart(self,):                
        circ_adj = self.getRealPart_base_circ(self.nspins, self.circ, self.Q, self.j_ref, self.gamma)
        #################### for each j2  The T_{j_ref -> j2} is different 
        indexs = self.df_ampl.index  ### the observed bit strings from shots; j's
        parts_real= []# [[ 0]]  # ref has c_j_ref = 0 + 0j
        part_indexs= [] #[self.j_ref]
        m1s = []
        c2s = []
        
        for j2 in indexs:
            circ_uhu_adj = self.getRealPart_ref_circ(self.j_ref, j2, self.nspins,  circ_adj)
            ####  from shots
            m_ref = computeAmplFromShots(circ_uhu_adj, self.shots, self.j_ref)
            ### collect results            
            m1s.append(m_ref)
            ####
            m_ref = np.round(m_ref, self.significant_figure)
                                       
            #### amplitude  from shots
            c2 = self.df_ampl['|c_j|'][j2]
            ### collect results
            c2s.append(c2)
            #### 
            c2_2 = c2**2 ### |c_j|^2
            self.c2_2    = np.round(c2_2, self.significant_figure)
            
            
            
            
            #### compute the cos of theta        
            real_part = (m_ref - (1/4) * self.c2_2 *\
                          (np.cos(self.gamma/2)**2) - (1/4)*(np.sin(self.gamma/2))**2 )/\
                            ((-1/2) * np.cos(self.gamma/2) * np.sin(self.gamma/2))
            
            #### round to allowed prcision
            real_part = np.round(real_part, self.significant_figure)
               
            ### collect results
            parts_real.append([ real_part ])
            part_indexs.append(j2)
           
        ### final results
        parts_real = pd.DataFrame(parts_real, index= part_indexs).round(self.significant_figure)        
        parts_real.columns=[ 'c_real_sim' ]   
        
        ###
        m1s = pd.DataFrame(m1s, index= part_indexs)
        m1s.columns=[ 'm1s-real-sim' ]  
        c2s = pd.DataFrame(c2s, index= part_indexs)       
        c2s.columns=[ 'c2s-real--sim' ]  
        
        ### attribute to the obj
        self.parts_real =  parts_real
        self.parts_real_m1s =  m1s
        self.parts_real_c2s =  c2s
        
    ######## TOOLKIT
    @staticmethod
    def computeAmplFromShots(circ_uhu_adj, shots, j_ref):
        return computeAmplFromShots(circ_uhu_adj, shots, j_ref)
    
    
    @staticmethod
    def getImagPart_base_circ(nspins, circ_U , Q, j_ref,  gamma =   np.pi/10):
        
        circ_adj = QuantumCircuit(nspins+1)
        
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
        
        for (q,o) in enumerate(J1):
            if o==1:            
                circ_adj.cx(nspins, q)                           
        circ_adj.x(nspins)  
        
        
        
    @staticmethod
    def getImagPart_ref_circ(j_ref, j2,nspins,  circ_adj):
        
        #### T Gate        
        p_12_int = j2^j_ref                
        ## operator
        P_12 = getBinary(p_12_int, nspins).tolist()+[0] #bitstring array of p12
        mult_gate, op_count = multiqubit(P_12, np.pi/4) # turned into T gate
        circ_uhu_adj = circ_adj.compose( mult_gate ) #add to the circuit
        
        return circ_uhu_adj
    
    
    
    @staticmethod  
    def getRealPart_ref_circ(j_ref, j2,nspins,  circ_adj):
        
        #### T Gate        
        p_12_int = j2^j_ref                
        ## operator
        P_12 = getBinary(p_12_int, nspins).tolist()+[0] #bitstring array of p12
        mult_gate, op_count = multiqubit(P_12, np.pi/4) # turned into T gate
        circ_uhu_adj = circ_adj.compose( mult_gate ) #add to the circuit
        
        return circ_uhu_adj
    
    
    @staticmethod
    def getRealPart_base_circ(nspins, circ_U , Q, j_ref,  gamma =   np.pi/10):


        circ_adj = QuantumCircuit(nspins+1)
        
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
        
        for (q,o) in enumerate(J1):
            if o==1:            
                circ_adj.cx(nspins, q)                           
        circ_adj.x(nspins)  
        
        
        ### S and H on ancillary
        circ_adj.s(-1)  
        circ_adj.h(nspins)

        return circ_adj