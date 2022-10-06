#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 19:27:20 2022

@author: pej

"""


import numpy as np
import pandas as pd
from BasicFunctions.functions import getQCirc, getUQUCirc
from Amplitude.computeAmplitudeV1 import computeAmplitude
from Phase.computePhaseStateVec_test import computePhaseStateVec
from BasicFunctions.functions import timing


# @timing
def findCircuitParameters(circ,
                          iQ, 
                          Q,
                          circ_UQU,                 
                          delta_w, 
                          eta,                    
                          shots, 
                          nspins,                                                                       
                          significant_figures
                          ):    
    
    
    ### returns df_comp_bm, df_ampl, m_support, std_prob, 
    ### drop_in_peak, df_ampl_org
    df_comp_bm, df_ampl, m_support,\
    std_prob,  drop_in_peak, df_ampl_org = computeAmplitude(circ_UQU,  shots, 
                                                   eta, significant_figures)
    
    print('eta = ', eta , end= ' ---   ') 
    print('significant_figures = ', significant_figures , end= ' ---   ')
    print('iQ = ' , iQ, '  len(list_js) = ', df_ampl_org.shape[0])
    
    print('df_ampl.shape[0] = ' , df_ampl.shape[0])
    print('df_ampl_org.shape[0] = ' , df_ampl_org.shape[0])
    
    ### if nothing observed 
    if df_ampl.empty==True:
        return [[], [], m_support]
    
    ### if observed 
    else:
        ### if nothing observed 
        df_cj = computePhaseStateVec(  df_ampl,  
                                      df_comp_bm, 
                                      circ,
                                      Q,
                                      circ_UQU, 
                                      significant_figures, 
                                      nspins)            

        ### if at least one y_j is non-zero
        try:                        
            try:
                Q_expect = df_cj.loc[0][0].real
            except:
                Q_expect = 0 
            
            ### norm    
            norm = 1- 2*delta_w * Q_expect + delta_w ** 2
            
            ### y_j
            df_cj= df_cj.loc[df_ampl.index]            
            df_y_j =  ((df_cj)[0] * -(delta_w/norm) )                        
            df_y_j = pd.DataFrame(df_y_j, columns=[0])
            
            ### j_list and y_list
            j_list = df_y_j.loc[df_y_j[0]!=0].index
            df_y_j = df_y_j.loc[j_list]            
            # remove 0
            try:
                df_y_j = df_y_j.drop(index=0)
            except:
                pass            
            list_y_j = df_y_j[0].tolist() 
            list_j = df_y_j.index.tolist()
        ### if all y_j's are zero
        except:
            return [[], [], m_support]

        ### return successfully
        return [list_j, list_y_j, m_support, std_prob, drop_in_peak,  df_ampl_org]

def findCircuitParametersExt(inputs):
    '''
    

    Parameters
    ----------
    inputs : array
        array of the inputs to findComponnets

    Returns
    -------
    TYPE
        an array of the y_j and the |j> itself.

    '''
    circ = inputs[0]
    Q=inputs[1]
    delta_w=inputs[2]
    eta = inputs[3]    
    shots=inputs[4]
    nspins = inputs[5]
    iQ = inputs[6]    
    significant_digits = inputs[7]
    return findCircuitParameters(circ, 
                          Q,
                          delta_w, 
                          eta,                           
                          shots, 
                          nspins, 
                          iQ,  
                          significant_digits)







if __name__=='__main__':
    '''
    Try to catch scenarios where there are significant errors
    
    '''
    
    pass


    ### external library
    import numpy as np
    import pandas as pd    
    from qiskit import QuantumCircuit
    
    ### internal library
    from BasicFunctions.functions import getBinary, getState
    
    ### inputs 
    nspins = 10
    n_h = nspins
    ngates = nspins
    num_layers = 6
    num_itr =3        
    machine_precision = 10        
    shots = 10000
    eta =10
    significant_figures=3
    iQ = 0
    delta_w = 0.01
    ###### seed for random
    seed = 365
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
    
    ### a random Q
    Q = getRandomQ(nspins)    
    
    ### circ U^ Q U
    circ_UQU = getCircUQU(circ, Q)
    
    #### find { (j, y_j) } <--> list_j, list_y_j
    res = findCircuitParameters(circ,
                         iQ, Q,
                         circ_UQU,                 
                         delta_w, 
                         eta,                    
                         shots, 
                         nspins,                                                                       
                         significant_figures
                         )
    
    print('list_j')
    print(res[0])
    
    print('list_yj')
    print(np.round(res[1], significant_figures) )
    
    













