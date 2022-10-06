#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:06:53 2022

@author: pej
"""

import os
import numpy as np
import pandas as pd
import time
from numpy import linalg as lg

from Problems.getExactHamilt import constructExactHamiltonian


def sortByWs(hs, ws,  precision = 0):
    hs_apprx = []
    ws_apprx = []
    ws2 = [w**2 for w in np.array(ws)]
    for (i,w2) in enumerate(ws2):
        if np.round(np.sqrt(w2), precision)==0:
            pass
        else:
            hs_apprx.append(hs[i])
            ws_apprx.append(ws[i])

    return hs_apprx, ws_apprx

#####################################
def getNuclearExample(file_name):
    hs, ws, dt = getNuclearExample_master(file_name)
    ##########################################
    return hs, ws
#####################################
def getNuclearExample_master(file_name):
    file_name = os.path.dirname(__file__) + '/' + file_name
    print(file_name)
    dt = pd.read_table(file_name)
    dt.columns=[0]
    dt_clean = pd.DataFrame()
    for row in range(dt[0].shape[0]):
        w, h = dt[0][row].split()
        P = []
        for O in list(h):
            if O=='I':
                P.append(0)
            if O=='X':
                P.append(1)  
            if O=='Y':
                P.append(2)
            if O=='Z':
                P.append(3)
                
        dt_clean = pd.concat((dt_clean, pd.DataFrame([float(w), P]).T))
        
    dt_clean.index = dt.index
    hs = dt_clean[1].tolist()
    ws = dt_clean[0].tolist()
    
    ##########################################
    return hs, ws, dt_clean


if __name__=='__main__':
    filename = 'ham-JW-full.txt'
    hs, ws = getNuclearExample(filename)
    
    hs, ws, dt =getNuclearExample_master(filename)
    
    
    
    #########################################################################
    #####################1 nuclear
    ################################################
    input_dir = '/../../Inputs/'
    out_put_dir = './../../ResultsV1/'
    results_folder  = str(int(time.time()))
    
    # ham-JW-full.txt ;
    # ham-0p-pn-spherical.txt ;
    # ham-0p-spherical.txt
    # ham-0p32.txt
    file_name = 'ham-0p-spherical.txt'
    
    
    hs_, ws_ = getNuclearExample(file_name)
    ## initial state     
    hs = []
    ws = []
    for (i, w) in enumerate(ws_):
        if w!=0:
            hs.append(hs_[i])
            ws.append(ws_[i])
    ### nspins
    nspins = len(hs[0])
    
    
    # ########## Hamiltonian matrix and  bm    
    H_mtx_file = 'Hamilt_'+file_name[:-4]   
    path = os.path.dirname(__file__)+input_dir+ H_mtx_file 
    
    
    if os.path.exists(path+'.npy'):
        print('H exists ')
        H  = np.load(path+'.npy')
        H = np.matrix(H)
    else:
        t0 = time.perf_counter()    
        H  = constructExactHamiltonian(hs, ws)  
        t1 = time.perf_counter()
        print('constructHamilt time = ', t1-t0, '\n')
        t0 = time.perf_counter()        
        np.save(path, H)
        
    t0 = time.perf_counter()
    Eexct, Vexct = lg.eigh(H)
    t1 = time.perf_counter()
    print('Diagonalization performance time = ', np.round(t1-t0, 4))
    print(Eexct.round(4).real)
    
    
    