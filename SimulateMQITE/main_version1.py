#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:46:10 2022

@author: pejmanjouzdani
"""


from time import time  , perf_counter  , ctime
  
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import linalg as lg

from qiskit import QuantumCircuit
import qiskit.quantum_info as qi


##### Some core utilities
from BasicFunctions.functions import getExact, getQCirc, getITE, getNum_j_UHU
from BasicFunctions.functions import  getExpec, getUQUCirc, getGateStats
from BasicFunctions.functions import getState, normalize, getOverlap


from UpdateCircuit.findCircuitParametersV1 import findCircuitParameters
from UpdateCircuit.updateCirc import updateCirc


from Problems.spinProblems import RandomIsing, Ising, generateRandom_k_local_XZ



def main(num_itr, 
         shots,
         hs,
         ws,
         delta,
         psi_bm,
         hs_mtx,
         eta,
         significant_figures,
         H_mtx,
         bm_only,
         circ,
         machine_precision,                  
         n_H,
         E_exact,
         res,
         output_dir,
         results_folder,
         ):
    '''
    

    Parameters
    ----------
    num_itr : int
        num teration --> T in the paper.
    shots : int
        number of shots --> \chi in paper.
    hs : list
        list of Pauli strings -- > Qs in the paper.
        each Q (here denoted by h) is provided as a list compromised of {0,1,2,3}
        example Q=XXI <==> [1,1,0]
        
    ws : list
        list of float values, the weights of each Q_k --> w_k in paper
        
    delta : float
        the delta --> delta in the paper.
        
    psi_bm : numpy matrix 
        for benchmarking against simulation 
        .
    hs_mtx : list
        for benchmarking against simulation.
        
    eta : int
        the cut-off number of dominant componenets to retain.
        
    significant_figures : int
        specifies the accuracy/precision --> \epsilon in the paper.
        
    H_mtx : numpy matrix
        This is to obtain the observables --> line 21.
        The alternative is to sum expectations \sum_k <Q> w_k at each time step
        The latter is time consumuing but trivial.
        
    bm_only : bool
        specifies whether to do just bm (bm_only=True) or both sim and bm.
        
    circ : qiskit circuit
        the initialized ciruict. We used circ=U == I --> (line 1)
        if a preparation is needed a different circ instruction must be 
        provided.
        
    machine_precision : int
        similar to significant_figures, but for a classical bm and 
        general assupmtion on how good our classical simulation is.
        
    n_H : int
        the number of Qs most likely not used anywhere in the code (?)
        
    E_exact : TYPE
        for bm. The exact energy.
        
    res : dict
        the results are collected and pushed into this dictionary. It is saved at 
        every time step at the location results_folder+output_dir.
        
    output_dir : str
        the sub folder of the results.
        if not exisiting on file, it will be created (mkdir).
        
    results_folder : str
        the main folder where all the results are stored.
        This must be existing on disk
    
    Returns
    -------
    res : dict
        the dictionary of all results.
        
        It contains dynamical and static results.
        
        Dynamical are the one that are collected at every time step or
        even at every time step for every Q.
        
        It also contains info of the overall simulation. like the max 
        number of max_m_support.

        To discover all the available keys. do a small test and map out 
        the tree structure        
        
    '''
    
    
    ####
    ##### static info
    nspins = len(hs[0])
    
    
    
    
    df_0 = pd.DataFrame([res]).T
    print(df_0)
    #########################################
    res['evolution']={}    
    
    ################## initial energy
    E_bm = getExpec(psi_bm, H_mtx) 
    e, v= lg.eigh(H_mtx)
    psi_gs = np.matrix(v[:, 0]).reshape(H_mtx.shape[0], 1)
    res['psi_exact']=psi_gs
    #################################
    res['evolution'][0]={}
    res['evolution'][0]['bm']={}
    res['evolution'][0]['bm']['E_bm']=E_bm
    res['evolution'][0]['sim']={}
    res['evolution'][0]['sim']['E_sim']=E_bm
    res['evolution'][0]['bm']['psi_bm']=psi_bm
    res['evolution'][0]['sim']['psi_sim']=psi_bm 
    res['evolution'][0]['sim']['circ']=circ
    res['evolution'][0]['sim']['total_gate_increment']=0  
    
    #### for ploting as we simulate (terminal)
    E_bm_arr = [E_bm]
    E_sim_arr = [E_bm]
    
    
    ##################
    U_0 = np.identity(H_mtx.shape[0], dtype=complex)
    num_j_UHU_t = getNum_j_UHU(psi_bm, H_mtx, U_0, significant_figures)
    print('num_j_UHU_t = ', num_j_UHU_t)    
    num_j_UHU_t_arr = [num_j_UHU_t]
    
    ##################
    overlap1 = getOverlap(psi_bm, psi_gs)                
    fidelity_exact_sim = np.sqrt((overlap1*overlap1).real) 
    print('fidelity_exact_sim = ', fidelity_exact_sim)
    
    ##################
    ### bm 2
    psi_bm_H = np.matrix.copy(psi_bm)
    
    ##################
    res['evolution'][0]['sim']['num_j_UHU_t']=num_j_UHU_t
    res['evolution'][0]['sim']['fidelity_exact_sim']=fidelity_exact_sim
    res['evolution'][0]['sim']['max_std_avg']=0
    res['evolution'][0]['sim']['drop_in_peak_last']=1 # since it starts at |0>
    
    ##################
    res['evolution']['max_m_support'] = 0
    res['evolution']['max_j_support'] = 0 
    max_j_support_arr = [res['evolution']['max_j_support']]
    ##################
    
    ##################
    total_number_of_gates = 0
    stds_avg_arr=[0]
    drop_in_peak_last_arr=[1]
    
    ##################
    for t in range(1, num_itr+1):
        print('------------')
        print('t = ', t )  
        print('time = ', t*delta )  
        print('shots = ' , shots)

        ##################
        res['evolution'][t]={}    
        res['evolution'][t]['sim']={} 
        res['evolution'][t]['bm']={}
        res['evolution'][t]['sim']['res']={}
        
        ##################
        #### for timing
        time1= perf_counter()
        print('------   > ' , ctime())
        
        ##################
        res['evolution'][t]['sim']['start_time']=ctime()  
        
        ##################
        total_gate_increment = 0
        stds_prob_Q = []
        
        ##################
        ### another bm
        psi_bm_H = psi_bm_H - delta * (H_mtx.dot(psi_bm_H))
        psi_bm_H = normalize(psi_bm_H)            
        psi_bm_H = np.matrix(psi_bm_H)   
        E_bm_H = getExpec(psi_bm_H, H_mtx) 
        
        #################
        res['evolution'][t]['bm']['E_bm_H']=E_bm_H
        
        ##################
        for ih in range(len(hs_mtx)):  
            
            delta_ = delta * ws[ih]
            ##################  bm
            psi_bm_ = psi_bm - delta_ * (hs_mtx[ih].dot(psi_bm))
            # psi_bm_ += + (delta_**2)/2  * hs_mtx[ih].dot((hs_mtx[ih].dot(psi_bm)))
            psi_bm_ = normalize(psi_bm_)            
            psi_bm = np.matrix(psi_bm_)            
            E_bm = getExpec(psi_bm, H_mtx)  
            
            ##################
            res['evolution'][t]['bm']['E_bm']=E_bm
            res['evolution'][t]['bm']['psi_bm']=psi_bm
            
        
        ##################
        for (ih, h) in enumerate(hs):            
            delta_ = delta * ws[ih]
            
            ##################
            if bm_only==True:
                pass
            else:
                
                ##################
                res['evolution'][t]['sim']['res'][ih]={}
                
                ##################
                circ_h = getQCirc(circ, h)                
                circ_uhu = getUQUCirc(circ, circ_h)
                
                ##################                
                [j_indxs, y_js, m_support, std_prob_Q,drop_in_peak, 
                 df_ampl_org] = findCircuitParameters(circ,
                                                      ih,
                                                      h,
                                                      circ_uhu,               
                                                      delta_, 
                                                      eta,                    
                                                      shots, 
                                                      nspins,                                                                                                   
                                                      significant_figures
                                                      )
                
                ##################
                circ_new , multigate_gate_stat = updateCirc(j_indxs, y_js, circ, )  
                ##################
                
                ##################
                state_new = getState(circ_new, machine_precision)                                                              
                ##################
                
                ##################
                circ = QuantumCircuit.copy(circ_new)                                
                ##################
                
                ################## oubservables
                overlap2 = getOverlap(state_new, psi_bm)
                fidelity_bm_sim = np.sqrt((overlap2*overlap2.conjugate()).real)   
                
                overlap1 = getOverlap(state_new, psi_gs)                    
                fidelity_exact_sim = np.sqrt((overlap1*overlap1.conjugate()).real)   
                
                E_sim = getExpec(state_new, H_mtx) 
                ##################
                
                
                ##################
                res['evolution'][t]['sim']['res'][ih]['js']=j_indxs
                res['evolution'][t]['sim']['res'][ih]['y_js']=y_js
                res['evolution'][t]['sim']['res'][ih]['Q'] = h
                res['evolution'][t]['sim']['res'][ih]['w_Q'] = delta_
                res['evolution'][t]['sim']['res'][ih]['w'] = ws[ih]
                res['evolution'][t]['sim']['res'][ih]['std_Q'] = std_prob_Q
                res['evolution'][t]['sim']['res'][ih]['drop_in_peak'] = drop_in_peak
                ##################                
                stds_prob_Q.append(std_prob_Q)
                
                
                
                
                ##################
                res['evolution'][t]['sim']['res'][ih]['sig-digits'] = significant_figures
                res['evolution'][t]['sim']['res'][ih]['num_j']=len(y_js)
                res['evolution'][t]['sim']['res'][ih]['fidelity_bm_sim'] = fidelity_bm_sim
                res['evolution']['max_m_support']  = max(m_support,  
                                         res['evolution']['max_m_support'] )
                res['evolution']['max_j_support']  = max(len(j_indxs),  
                                         res['evolution']['max_j_support'] )
                
                ##################
                stats_Q = getGateStats(multigate_gate_stat, h, ih)                          
                ##################
                
                ##################
                res['evolution'][t]['sim']['res'][ih]['num_gates']=stats_Q['total'].values[0]
                res['evolution'][t]['sim']['res'][ih]['gates-stats']=stats_Q
                
                ##################
                total_gate_increment += stats_Q['total'].values[0]
                ##################
                
                
                
        print()        
        ################## record ending time               
        time2= perf_counter()            
        print('------------- > Performance time :  ', 
              np.round(time2-time1, 4))
        ##################
        
        
        
        ##################
        if bm_only==True:
            print('E_bm = ', E_bm.round(5) )      
            print('E_bm_H = ', E_bm_H.round(5) )      
            E_bm_arr.append(E_bm)
            # ##################
            plt.plot(psi_bm.real.round(3), 'o')            
            plt.plot(psi_gs.real.round(3), '*')
            plt.show()
            plt.close()
            plt.plot(np.array(E_bm_arr) - E_exact, '-o')                        
            plt.show()
            plt.close()
            
            
        else:
            
            ##################
            res['evolution'][t]['sim']['performance-time']=np.round(time2-time1, 4)
            res['evolution'][t]['sim']['E_sim']=E_sim
            res['evolution'][t]['sim']['psi_sim']=state_new
            res['evolution'][t]['sim']['circ']=circ
            res['evolution'][t]['bm']['psi_bm']=psi_bm
            res['evolution'][t]['sim']['total_gate_increment']=total_gate_increment
            res['evolution'][t]['sim']['fidelity_bm_sim']=fidelity_bm_sim
            res['evolution'][t]['sim']['fidelity_exact_sim']=fidelity_exact_sim
            res['evolution'][t]['sim']['max_std_avg']=np.average(stds_prob_Q)
            res['evolution'][t]['sim']['drop_in_peak_last']=drop_in_peak
            res['evolution'][t]['sim']['df_ampl_org']=df_ampl_org
            
            ##################
            print('max_std_avg = ', np.average(stds_prob_Q))   
            print('drop_in_peak_last = ', drop_in_peak)   
            
            stds_avg_arr.append(         np.average(stds_prob_Q))
            drop_in_peak_last_arr.append(drop_in_peak) # last observed
            
            ##################  
            U_t = np.matrix(qi.Operator(circ).data)
            num_j_UHU_t = getNum_j_UHU(state_new,H_mtx, U_t, 
                                       significant_figures)
            print('num_j_UHU_t = ', num_j_UHU_t)            
            num_j_UHU_t_arr.append(num_j_UHU_t)
            
            res['evolution'][t]['sim']['max_std_avg']
            
            ##################
            res['evolution'][t]['sim']['num_j_UHU_t']=num_j_UHU_t
            
            ##################
            max_j_support_arr.append(res['evolution']['max_j_support'] )
             
            # ##################
            plt.plot(state_new.real.round(3), 'x', label=' circ state (real)')            
            # plt.plot(psi_gs.real.round(3), '*', label=' exact gs state (real)')
            plt.plot(psi_bm.real.round(3), 'o', label=' bm (ITE) state (real)')
            plt.legend()
            plt.show()
            plt.close()

            plt.plot(state_new.imag.round(3), 'x', label=' circ state (imag)')            
            # plt.plot(psi_gs.imag.round(3), '*', label=' exact gs state (imag)')
            plt.plot(psi_bm.imag.round(3), 'o', label=' bm (ITE) state (imag)')
            plt.legend()
            plt.show()
            plt.close()

            E_bm_arr.append(E_bm)
            E_sim_arr.append(E_sim)
            plt.plot(np.array(E_bm_arr) , '-o', label='E-E-axact (bm)')                        
            plt.plot(np.array(E_sim_arr) , '-x', label='E-E-axact  (sim)')    
            # plt.yscale('log')
            plt.grid(axis='y')                    
            plt.show()
            plt.close()
            
            
            # plt.plot(num_j_UHU_t_arr, '-o')            
            # plt.show()
            # plt.close()
            # plt.plot( stds_avg_arr, '-o')            
            # plt.show()
            # plt.close()
            # plt.plot( drop_in_peak_last_arr, '-o')            
            # plt.show()
            # plt.close()
            # plt.bar( df_ampl_org.index, df_ampl_org.values.T.tolist()[0])            
            # plt.show()
            # plt.close()
            
            
            ##################
            print('max_m_support / 2**nspins= ' , 
                  res['evolution']['max_m_support']/2**nspins)
            print('fidelity_bm_sim = ' , fidelity_bm_sim)
            print('fidelity_exact_sim = ' , fidelity_exact_sim)
            print('total_gate_increment = ' , total_gate_increment)    
            
            ##################
            total_number_of_gates+=total_gate_increment
            print('total_number_of_gates = ' , total_number_of_gates)    
            
            ##################
            ops_count = circ.decompose().count_ops()
            ops_count=pd.DataFrame([ops_count])
            print('total_number_of_gates circ', ops_count.sum().values[0])            
            
            ##################
            print('E_bm_H = ', E_bm_H.round(5) )
            print('E_bm = ', E_bm.round(5) )
            print('E_sim = ', E_sim.round(5))        
        
        ################
        if bm_only==True:
            ##################
            print('E_exact ', E_exact)
            print('E_bm - E_exact', E_bm - E_exact)
            print('E_bm/E_exact', np.round(E_bm/E_exact, 6) )
            
            
            
        else:
            ##################
            print('E_exact ', E_exact)
            print('(E_bm - E_exact)/E_exact', (E_bm - E_exact)/ E_exact ) 
            print('(E_sim - E_exact)/E_exact', (E_sim - E_exact)/ E_exact )
            print('E_sim/E_exact', np.round(E_sim/E_exact, 6) )
    
        
    ##################
        filename =output_dir+results_folder+'/results_all.pkl'
        try:
            os.remove(filename)
        except:
            pass
        with open(filename, 'wb') as f:
            pickle.dump(res, f)
        f.close()
    
    ##################
    return res
            


if __name__=='__main__':
    inputs = {}
    
    #########################################################################
    #####################1 nuclear
    ################################################
    # input_dir = '/../../Inputs/'
    output_dir = './../Results/'
    inputs['output_dir'] = output_dir
    
    results_folder  = str(int(time()))
    inputs['results_folder'] = results_folder
    
    
    try:
        os.mkdir(output_dir)
    except:
        pass
    os.mkdir(output_dir +results_folder)
    ###############
    
    
    ###########  inputs         
    this_file = __file__
    inputs['this_file'] = this_file
    
    
    
    ###############
    seed = 12321
    inputs['seed'] = seed
    
    nspins = 6
    inputs['nspins'] = nspins
    
    machine_precision = 10
    inputs['machine_precision'] = machine_precision
    
    delta = 0.1
    inputs['delta'] = delta
    
    
    num_itr = 30 #int(  1 *  nspins/delta)
    inputs['num_itr'] = num_itr
    
    ###########
    shots = 1000
    inputs['shots']=shots
    
    eta = 10
    inputs['shots']=eta
    # add this to the sig-digits assumed from shots
    # We suppose the ampl can be obtained upto this precision
    significant_figures =2
    inputs['significant_figures_add'] = significant_figures    
    
    ############   
    
    W_min = 0
    inputs['W_min'] = W_min
    
    W_max = 1
    inputs['W_max'] = W_max
    
    Hz_min = -1.00
    inputs['Hz_min'] = Hz_min
    
    Hz_max = -1.00
    inputs['Hz_max'] = Hz_max
    
    
    k = 3
    inputs['k'] = k
    
    
    bm_only = False
    inputs['bm_only'] = bm_only
    
    ########  seed random
    np.random.seed(seed)
    
    # ######## Hamiltonin = H = \sum_k w_k * h_k   
    # n_h = int(nspins*1.5)
    # inputs['n_h'] = n_h
    
    circ = QuantumCircuit(nspins)
    
    #### change the initial state    
    # circ.h(0)
    # for q in range(1,nspins):
    #     circ.cnot(q-1, q)
    
    
    
    psi_bm = getState(circ, machine_precision)
    
    
    
    # ###################################################################
    # ###################
    # r = Ising(nspins)
    # hs = r.oneDTIM()
    # inputs['hs-function'] = r.oneDTIM.__name__
    # ws = np.random.uniform(W_min , W_max, size=nspins-1 ).tolist()  
    # ws = ws + np.random.uniform(Hz_min , Hz_max, size=nspins ).tolist() 
    
    
    # ###################  TISM on G
    # r = RandomIsing(nspins,k)
    # hs = r.getRandH_XX_Z()    
    # inputs['hs-function'] = r.getRandH_XX_Z.__name__
    # ws = np.random.uniform(W_min , W_max, size=len(hs)- nspins ).tolist() 
    # ws = ws + np.random.uniform(Hz_min , Hz_max, size=nspins ).tolist() 
    
    ##################  MAX CUT on G
    r = RandomIsing(nspins,k)
    hs = r.getRandH_ZZ()    
    inputs['hs-function'] = r.getRandH_ZZ.__name__
    ws = np.random.uniform(W_min , W_max, size=len(hs))
    ws = np.round(ws, 3)        
    
    
    # ################### K local on G with X and Z
    # n_h = nspins
    # hs = generateRandom_k_local_XZ(nspins, n_h, k)    
    # inputs['hs-function'] = generateRandom_k_local_XZ.__name__    
    # ws = np.random.uniform(W_min , W_max, size=len(hs))    
    # #################################################################
    
    
    # # #################### K local on G with X and Z
    # n_h = int(nspins*1)
    # hs = generateRandom_k_local_XYZ(nspins, n_h , k)
    # inputs['hs-function'] = generateRandom_k_local_XYZ.__name__    
    # ws = np.random.uniform(W_max , W_max, size=len(hs))    
    # ###################################################################
    
    
    
    ##### save ws
    inputs['ws'] = ws
    
    ##### save n_H
    n_H = len(hs)
    inputs['n_H'] = n_H
    
    
    ####### exact        
    hs_mtx, H_mtx, E_exact = getExact(hs, ws)    
    ####
    E_exact = E_exact    
    
    inputs['E_exact']=E_exact
    
    df_0 = pd.DataFrame([inputs]).T
    print(df_0)
    
    ########################
    res={}
    res['nspins']=nspins
    res['output_dir']=output_dir
    res['out-folder']=results_folder
    res['shots']=shots    
    res['eta']=eta
    res['significant_figures']=significant_figures
    res['delta']=delta
    res['machine_precision'] = machine_precision
    res['num_itr'] = num_itr
    res['E_exact']=E_exact
    res['n_H'] = n_H
    res['ws'] = ws
    res['hs-function'] = inputs['hs-function']
    res['bm_only'] = bm_only
    res['W_min'] = W_min
    res['W_max'] = W_max
    res['Hz_min'] = Hz_min
    res['Hz_max'] = Hz_max
    res['k'] = k
    res['seed'] = seed
    
    res['comment'] = "Max CUT for k=3 weighted; W=[0-1]BUT with a different initial state; circ.rx(np.pi/3, q) ; nspins=10, I use exact phase"
    
    
    results = main(
        num_itr, 
          shots,
          hs,
          ws,
          delta,
          psi_bm,
          hs_mtx,
          eta,
          significant_figures,
          H_mtx,
          bm_only,
          circ,
          machine_precision,                   
          n_H,
          E_exact,
          res,
          output_dir,
          results_folder,
        )
    