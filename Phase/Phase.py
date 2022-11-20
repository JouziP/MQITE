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
                  df_ampl,
                  nspins, 
                  circ_U, 
                  Q,
                  significant_figures, 
                  shots,
                  machine_precision=10,
                  j_ref=None,
                  gamma= np.pi/10):
        
        #### static
        # self.amplObj            = amplObj
        self.nspins             = nspins
        self.circ_U               = circ_U ## the U_t circuit
        self.Q                  = Q # the Q Pauli string given in form of an array of 0,...,3
        self.significant_figures = significant_figures # significant_figures
        self.shots              = shots # 
        self.machine_precision  = machine_precision        
        self.df_ampl            = df_ampl# |c_j| in U^QU|0>=\sum_j c_j|j>
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
            Q_expect = self.df_ampl.loc[0]['|c_j|'].real
        except:
            Q_expect = 0 
        
        ### norm    
        norm = 1- 2*delta * Q_expect + delta ** 2
        
        
        ### to avoid devision by zero error
        _epsilon = 10**(-1 * 2 * self.significant_figures)
        
        ### the actual parameters        y_j
        c_j_times_delta_real = self.c_real * -(   delta  /   (norm + _epsilon)  ) 
        c_j_times_delta_imag = self.c_imag * -(   delta  /   (norm + _epsilon)  )         
        params_real_plus_imag = -c_j_times_delta_real.values + 1j*c_j_times_delta_imag.values
        ## -y_j^(re) + 1j * y_j^(im)
        self.yj_parameters =  pd.DataFrame(params_real_plus_imag, columns=['y_j'], index = self.c_real.index) ## -y_j^(r)
        
        ### delta * cj the otput that will be passed to update circuit
        delta_times_cj  = c_j_times_delta_real.values + 1j*c_j_times_delta_imag.values
        ## delta * ( c_j ) 
        self.delta_times_cj =  pd.DataFrame(delta_times_cj, columns=['delta_times_cj'], index = self.c_real.index)
        
        # remove 0 --> double check
        try:
            self.delta_times_cj = self.delta_times_cj.drop(index=0)
        except:
            pass  
        
        ### j_list and delta_times_cj_values
        j_list = self.delta_times_cj.loc[self.delta_times_cj['delta_times_cj']!=0].index
        delta_times_cj_values = self.delta_times_cj.loc[j_list]      
        
                 
        #### returns list of js and list of delta*cj's
        return [
                    j_list.tolist(),
                    delta_times_cj_values.values.T.tolist()[0]
                ]
        
    
    
    ##### Q. COMPUTE  IMAG PART OF C_J
    def getImagPart(self,):
        
        circ_adj_for_imag_part = self.getImagPart_base_circ(self.nspins, self.circ_U, self.Q, self.j_ref, self.gamma)
        
        #################### for each j2  The T_{j_ref -> j2} is different 
        indexs = self.df_ampl.index  ### the observed bit strings from shots; j's
        
        m1s        = self.getMsImag(indexs, self.j_ref, self.nspins, 
                                    circ_adj_for_imag_part, 
                                self.shots, self.significant_figures )
        
        c2_2s        = self.getC2s(indexs, self.df_ampl, self.significant_figures)
        
        c_imag = self.getComponent(c2_2s, m1s, self.gamma)
        
        c_imag = pd.DataFrame(c_imag, index= indexs).round(self.significant_figures)   
       
        c_imag.columns=[ 'c_imag_sim' ]   
        
        # ###
        m1s = pd.DataFrame(m1s, index= indexs)
        m1s.columns=[ 'm1s-imag-sim' ]  
        
        c2_2s = pd.DataFrame(c2_2s, index= indexs)       
        c2_2s.columns=[ 'c2^2' ]  
        
        ### attribute to the obj
        self.c_imag =  c_imag
        self.m1s_from_imag =  m1s
        self.c2_2power2__imag =  c2_2s
        
    
    ##### Q. COMPUTE  REAL PART OF C_J
    def getRealPart(self,):                
        circ_adj_for_real_part = self.getRealPart_base_circ(self.nspins, self.circ_U, self.Q, self.j_ref, self.gamma)
        #################### for each j2  The T_{j_ref -> j2} is different 
        indexs = self.df_ampl.index  ### the observed bit strings from shots; j's
        
        m1s          = self.getMsReal(indexs, self.j_ref, self.nspins, 
                                      circ_adj_for_real_part, 
                                      self.shots, self.significant_figures )
        c2_2s        = self.getC2s(indexs, self.df_ampl, self.significant_figures)
        c_real   = self.getComponent(c2_2s, m1s, self.gamma)
        c_real   = pd.DataFrame(c_real, index= indexs).round(self.significant_figures)   
       
        c_real.columns=[ 'c_real_sim' ]   
        
        # ###
        m1s = pd.DataFrame(m1s, index= indexs)
        m1s.columns=[ 'm1s-real-sim' ]  
        
        c2_2s = pd.DataFrame(c2_2s, index= indexs)       
        c2_2s.columns=[ 'c2^2' ]  
        
        ### attribute to the obj
        self.c_real =  c_real
        self.m1s_from_real =  m1s
        self.c2_2power2__real =  c2_2s
        
        
        
        
    ########################################### TOOLKIT
    
    @staticmethod
    def getComponent(c2_2s, m1s, gamma):
        component = lambda m_ref, c2_2 : (m_ref - (1/4) * c2_2 *\
                      (np.cos(gamma/2)**2) - (1/4)*(np.sin(gamma/2))**2 )/\
                        ((-1/2) * np.cos(gamma/2) * np.sin(gamma/2))
        return list(map(component, m1s, c2_2s))
        
        
    @staticmethod
    def getC2s(indexs, df_ampl, significant_figures):
        c2_2s = []
        for j2 in indexs:
            #### amplitude  from shots
            c2 = df_ampl['|c_j|'][j2]            
            #### 
            c2_2 = c2**2 ### |c_j|^2
            c2_2    = np.round(c2_2, significant_figures)
            ### collect results
            c2_2s.append(c2_2)
        return c2_2s
    
    @staticmethod
    def getMsReal(indexs, j_ref, nspins, circ_adj, shots, significant_figures):
        m1s = []
        for j2 in indexs:
            circ_uhu_adj = Phase.getRealPart_ref_circ(j_ref, j2, nspins,  circ_adj)
            ####  from shots
            m_ref = computeAmplFromShots(circ_uhu_adj, shots, j_ref)            
            ####
            m_ref = np.round(m_ref, significant_figures)
            
            ### collect results            
            m1s.append(m_ref)
            
        return m1s
    
    @staticmethod
    def getMsImag(indexs, j_ref, nspins, circ_adj, shots, significant_figures):
        m1s = []
        for j2 in indexs:
            circ_uhu_adj = Phase.getImagPart_ref_circ(j_ref, j2, nspins,  circ_adj)
            ####  from shots
            m_ref = computeAmplFromShots(circ_uhu_adj, shots, j_ref)            
            ####
            m_ref = np.round(m_ref, significant_figures)
            
            ### collect results            
            m1s.append(m_ref)
            
        return m1s
    
    
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
        
        ### H on ancillary
        circ_adj.h(nspins)
        
        return circ_adj
        
        
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
    def getRealPart_ref_circ(j_ref, j2,  nspins,  circ_adj):
        
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