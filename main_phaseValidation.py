#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised from QSim3 repo 

Started on Aug 10 

This is the code for experiment 1 in the paper

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



##### Some core utilities
from BasicFunctions.functions import getExact
from BasicFunctions.functions import getState




from Problems.spinProblems import generateRandom_k_local_XZ



from SimulateMQITE.main_version1 import main




if __name__=='__main__':
    
    ### for displaying in console at start
    inputs = {}
    
    
    ################################################    
    ### results    
    output_dir = './../Results/'
    inputs['output_dir'] = output_dir
    results_folder  = str(int(time()))
    inputs['results_folder'] = results_folder    
    try:
        os.mkdir(output_dir)
    except:
        pass
    os.mkdir(output_dir +results_folder)

    ### this file address
    this_file = __file__
    inputs['this_file'] = this_file
    
    
    
    ############################### inputs
    seed = 1245
    inputs['seed'] = seed
    
    ## set seed random
    np.random.seed(seed)
    
    nspins = 6
    inputs['nspins'] = nspins
    
    ##  classical machine precision
    machine_precision = 10 # the most ideal prcision (assumed) -- classical computer
    inputs['machine_precision'] = machine_precision
    
    ##  time steps 
    
    delta = 0.3
    inputs['delta'] = delta
    
    
    num_itr = 10 
    inputs['num_itr'] = num_itr
    
    
    ##  precision of the quantum computer
    d = 2 # d
    inputs['d']=d
    
    eta = int(nspins**d)  # eta    
    inputs['eta']=eta
    
    significant_figures = 4 #
    inputs['significant_figures']=significant_figures
    
    # chi
    shots = 1000000
    inputs['shots']=shots
    
    ## some of the parameters are irrelevant to this problem should be removed
    W_min = 0
    inputs['W_min'] = W_min
    
    W_max = 1
    inputs['W_max'] = W_max
    
    Hz_min = -1.0 
    inputs['Hz_min'] = Hz_min
    
    Hz_max = -1.0
    inputs['Hz_max'] = Hz_max
     
    k = 3
    inputs['k'] = k
    
    ## Quatnum simulation + classical, or classical only (bm)
    bm_only = False
    inputs['bm_only'] = bm_only
    
    
    
   
    ############################### initialize circuit 
    circ = QuantumCircuit(nspins)
 
    
    ## for benchmarking
    psi_bm = getState(circ, machine_precision)
    
    
    ############################### The problem {(w, Q)} => ws, hs
    ## here Qs is represented by hs
    n_h = nspins
    hs =generateRandom_k_local_XZ(nspins, n_h , k)    
    ws = np.random.uniform(W_min , W_max, size=n_h ).tolist()      
    
    ## used in the paper (on case the random numbers are different)
    ws = [0.96051497, 
          0.85292819,
          0.13676468, 
          0.98014726, 
          0.71214027, 
          0.96220233]
    
    ##### function name
    inputs['hs-function'] = generateRandom_k_local_XZ.__name__    

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
    res['comment'] = "Experiment 1 in the paper"
    
    
    
    #########################
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
    