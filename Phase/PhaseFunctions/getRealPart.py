#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:22:28 2022
Created on Thu Oct  13 10:29:00 2022

@author: pej
"""



      
import numpy as np
import pandas as pd



from qiskit import QuantumCircuit
from MultiQubitGate.functions import multiqubit


from BasicFunctions.functions import getBinary, getState
from Phase.PhaseFunctions.computeAmplFromShots import computeAmplFromShots
        
####################################################################
####################################################################
####################################################################
####################################################################
################### When c0 == 0

def getRealPart(df_ampl, ### amplitude |c_j| computed from shots                 
                circ_U, 
                circ_UQU,
                Q,
                significant_figure, 
                nspins, 
                shots,
                j_ref, 
                machine_precision=10,                
                ): 
    '''
    for the scenario where j_ref=0 is NOT in df_ampl.index
    or c_0 == 0
    '''
    
    # ################## FOR COMPARISON
    # #### df_comp_exact, ### this is exact c_j for benchmark             
    # circ_state = getState(circ_UQU, machine_precision)
    # vec=circ_state[df_ampl.index, : ]
    # df_comp_exact = pd.DataFrame(vec, df_ampl.index)    
    # ################## 
    
    circ_adj = QuantumCircuit(nspins+1)
    _gamma =   -np.pi/4
    circ_adj.ry(_gamma, qubit=-1)  ### R_gamma
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
    
    
    
    #################### for each j2  The T_{j_ref -> j2} is different 
    indexs = df_ampl.index  ### the observed bit strings from shots; j's
    parts_real= [[ 0]]  # ref
    part_indexs=[j_ref]

    
    for j2 in indexs:
        
        #### T Gate        
        p_12_int = j2^j_ref                
        ## operator
        P_12 = getBinary(p_12_int, nspins).tolist()+[0] #bitstring array of p12
        mult_gate, op_count = multiqubit(P_12, np.pi/4) # turned into T gate
        circ_uhu_adj = circ_adj.compose( mult_gate ) #add to the circuit
        
        #####  from shots
        m1, __ = computeAmplFromShots(circ_uhu_adj, shots, j_ref)
        m1 = np.round(m1, significant_figure)
        
        
       
                       
        #### amplitude  from shots
        c2_2 = df_ampl[0][j2]**2 ### |c_j|^2
        c2_2    = np.round(c2_2, significant_figure)
        
        
        
        
        #### compute the cos of theta        
        real_part = (m1 - (1/4) * c2_2**2 * (np.cos(_gamma/2)**2) - (1/4)*(np.sin(_gamma/2))**2 )/ ((-1/2) * np.cos(_gamma/2) * np.sin(_gamma/2))
        
        #### round to allowed prcision
        real_part = np.round(real_part, significant_figure)
           
        ### collect results
        parts_real.append([ real_part, 
                           # df_comp_exact[0][j2].real
                           ])
        part_indexs.append(j2)
     
        
    parts_real = pd.DataFrame(parts_real, index= part_indexs).round(significant_figure)
    
        
    parts_real.columns=[ 'c_real_sim', 
                        # 'c_real_exct'
                        ]    
    
    return parts_real













if __name__=='__main__':
    pass
    
    
    from Amplitude.Amplitude import Amplitude
    
    
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
    nspins = 6   # >=2
    num_layers =2
    num_itr =1
    machine_precision = 10        
    shots = 1000
    eta = 100
    significant_figures = 3#np.log10(np.sqrt(shots)).astype(int)
    
    circ_U = getRandomU(nspins, num_layers)    
    Q = getRandomQ(nspins)
    circ_UQU = getCircUQU(circ_U, Q)
    
    print('Q=', Q)
    
    ##    ### amplitude |c_j| computed from shots 
    localAmplitudeObj  =  Amplitude(circ_U, circ_UQU, shots, eta, significant_figures, machine_precision)
    
    localAmplitudeObj()
    
    df_ampl  = localAmplitudeObj.df_ampl
    
    ######## EXCLUDE index 0
    try:
        df_ampl = df_ampl.drop(index=0)
        
    except:
        pass
    
    print('df_ampl')
    print(df_ampl)    
    print()
    
    
    #### choose the ref 
    j_ref = np.random.randint(0, 2**nspins)
    while j_ref in df_ampl.index:
        j_ref = np.random.randint(0, 2**nspins)
        
    print('j_ref = ' , j_ref)
    print('number of js = ' , df_ampl.shape[0])
    parts_real = getRealPart(df_ampl, ### amplitude |c_j| computed from shots                 
                    circ_U, 
                    circ_UQU,
                    Q,
                    significant_figures, 
                    nspins, 
                    shots,
                    j_ref, 
                    )
        
    print(parts_real)

    # print('\n\n##### sign errors')
    # num_error = 0
    # for jj in  parts_real.index:
    #     if parts_real['c_real_exct'][jj]>=0 and  parts_real['c_real_sim'][jj]>=0:                     
    #         num_error+=  0
    #     else:
    #         if parts_real['c_real_exct'][jj]<0 and  parts_real['c_real_sim'][jj]<0:                     
    #             num_error+=  0
    #         else:
    #             print(parts_real['c_real_exct'][jj],   '      ', parts_real['c_real_sim'][jj])
    #             num_error+= 1
        
    # print('error %= ', 
    #       np.round(num_error/parts_real.shape[0]*100, 2), 
    #       'num incorrect = ', 
    #       num_error, 
    #       'total = ', 
    #       parts_real.shape[0]
    #       )      
    
    # print('\n\n##### L2 norm')
    # error_L2 = 0
    # for jj in  parts_real.index:
    #     diff = np.abs(parts_real['c_real_exct'][jj] -  parts_real['c_real_sim'][jj])
    #     error = diff/( + 10**(-1*significant_figures) + np.abs(parts_real['c_real_exct'][jj] ) ) 
    #     print(error)
    #     error_L2+=  error/parts_real.shape[0]
    # print('---')
    # print(error_L2)