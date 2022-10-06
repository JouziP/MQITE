#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:22:28 2022

@author: pej
"""



      
import numpy as np
import pandas as pd



from qiskit import QuantumCircuit
from MultiQubitGate.functions import multiqubit


from BasicFunctions.functions import getBinary, getState
        
####################################################################
####################################################################
####################################################################
####################################################################
################### When c0 == 0
####################################################################
def getCosPhasesNoC0(df_ampl, 
                     df_comp, ### this is exact c_j for benchmark    
                     circ, 
                     Q,
                     significant_digits, nspins, 
                     j_ref=0): 
    '''
    for the scenario where j_ref=0 is in df_ampl.index
    c0=0
    '''         
    
    
    circ_adj = QuantumCircuit(nspins+1)
    gamma = np.pi/2 
    circ_adj.ry(gamma, qubit=-1)
    circ_adj.x(qubit=-1)
        
    ### U
    circ_adj = circ_adj.compose(QuantumCircuit.copy(circ) )        
    
    ### control Q
    for (q,o) in enumerate(Q):
        if o==1:            
            circ_adj.cx(-1, q)
        if o==2:
            circ_adj.cy(-1, q)
        if o==3:
            circ_adj.cz(-1, q)
        
    ### U^
    circ_adj = circ_adj.compose(QuantumCircuit.copy(circ).inverse())
    
    ### P_{0 j_ref}
    circ_adj.x(qubit=-1)
    J1 = getBinary(j_ref, nspins) + [0]
    for (q,o) in enumerate(J1):
        if o==1:            
            circ_adj.cx(-1, q)      
    circ_adj.x(-1, q)  
    
    ### S H on ancillary
    circ_adj.s(-1)  
    circ_adj.h(-1)
     
    #################### for each j2  The T_{j_ref -> j2} is different     
    indexs = df_ampl.index
    cos_phases= [[1, 0, 0]]
    cos_indexs=[j_ref]
    
    for j2 in indexs:
        
        ####################### T Gate
        # print('j2  : ', j2)          
        ####
        p_12_int = j2^j_ref        
        ## operator
        P_12 = getBinary(p_12_int, nspins).tolist()+[0]
        mult_gate, op_count = multiqubit(P_12, np.pi/4)
        circ_uhu_adj = circ_adj.compose( mult_gate )
        circ_state_adj  = getState(circ_uhu_adj, significant_digits)
        
        
        ## m1
        ###################### Using the satte vector to read ampl of m1 !!!
        ###################### the alternative is to make shots and distill
        ###################### for the ones where ancillary is in state 0
        ###################### It would have been very computationally heavy
        #### ancila ==0
        circ_state_adj = circ_state_adj.reshape(2, circ_state_adj.shape[0]//2).T[:, 0]
        m1 = circ_state_adj[j_ref, 0]
        # print('sqrt(m1)  : ', m1)    
        m1 = (m1*m1.conjugate()).real.round(significant_digits)
        # print('m1  : ', m1)    
        ##
        
        ## c1 and c2        
        c2= df_comp[0][j2]        
        # print('c1  : ', gamma, )    
        # print('c2  : ', c2, ) 
        
        # ### amplitude from state_vector           
        # c2_2 = np.sqrt((c2*c2.conjugate()).real) #.round(significant_digits) 
                
        #### amplitude  from shots
        c2_2 = df_ampl[0][j2]
        
        # m2_repl = (c2_2**2 +(gamma/2)**2 - 2*c2.real * (gamma/2))/4
        # print('m1  expected: ', np.round(m2_repl, significant_digits) )     
        # print('2*c2.real * gamma: ', 2*c2.real * gamma/4)    
        # print('((gamma/2) * c2_2 * 1/2 ): ', np.round(((gamma/2) * c2_2 * 1/2 ), significant_digits ) )    
        # print(np.round( ( - (gamma/2)**2 - c2_2**2)/4, significant_digits) )
        # print('c2_2  : ', c2_2 )    
        
        #### compute the cos of theta
        cos_theta = -(m1 - (gamma/2)**2/4 - c2_2**2/4)/((gamma/2) * c2_2 * 1/2 + 10**-float(significant_digits**2))        
        val = (m1 - (1/4) * c2_2**2 * (np.cos(gamma/2)**2) - (1/4)*(np.sin(gamma/2))**2 )/ ((-1/2) * np.cos(gamma/2) * np.sin(gamma/2))        
        
        #### round to allowed prcision
        cos_theta = np.round(cos_theta, significant_digits)
        # print('cos_theta  : ', cos_theta )    
        
        #### check for divergence
        if cos_theta<-1:
            # print('----------------> ', cos_theta)
            cos_theta=-1
        if cos_theta>+1:
            # print('----------------> ', cos_theta)
            cos_theta=1      
            
        # print('c2.real replicated: ', 
               # np.round(cos_theta*c2_2 , 4) ) 
        # print('c2.real : ', np.round(c2.real, 4) ) 
        
        
        cos_phases.append([cos_theta,  val, c2.real])
        cos_indexs.append(j2)
        # print()
        # print()
        # print() 
        
    cos_phases = pd.DataFrame(cos_phases, index= cos_indexs).round(significant_digits)
    
    
    cos_phases.columns=['cos_theta', 'c_real_sim', 'c_real_exct']   
     
    return cos_phases













if __name__=='__main__':
    pass

    from matplotlib import pyplot as plt
    #### a random circ
    nspins = 10
    n_h = nspins
    ngates = nspins
    num_layers = 3
    ####
    machine_precision = 10            
    shots = 1000000
    eta = 100
    significant_figures = 3    

    ######
    seed = 66587
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
    
    
    #### choose the ref 
    j_ref = np.random.randint(0, 2**nspins)
    while j_ref in df_ampl.index:
        j_ref = np.random.randint(0, 2**nspins)
                  
    cos_phases = getCosPhasesNoC0(df_ampl, 
                                  df_comp_bm,
                                  circ, 
                                  Q,
                                  significant_figures, 
                                  nspins, 
                                  j_ref)
        
    print(cos_phases)

    ##### errors
    cos_indexs= cos_phases.index
    num_error=(np.sum(np.sign(cos_phases)['c_real_sim']!=np.sign(cos_phases)['c_real_exct']))
    print('error %= ', 
          np.round(num_error/len(cos_indexs)*100, 2)  ,        
          'num incorrect = ', 
          num_error, 
          'total = ', 
          len(cos_indexs)
          )  
    
    print('\n\n##### L2 norm')
    cos_indexs= cos_phases.index
    error=sum((np.sign(cos_phases)['c_real_sim'] - np.sign(cos_phases)['c_real_exct']).apply(lambda x: x**2))
    
    print('L2 norm/num componenets = ', 
          np.round(error/len(cos_indexs), 2),            
          ) 