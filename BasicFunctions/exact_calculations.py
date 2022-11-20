#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:43:09 2022

@author: pejmanjouzdani
"""



# from scipy.linalg import schur
from time import time  #  , perf_counter, ctime
# import os
import numpy as np
from numpy import linalg as lg
# from scipy.sparse.linalg import cg
# import pandas as pd
# from matplotlib import pyplot as plt
from functools import wraps




def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r --->  %2.4f sec' %(f.__name__,   te-ts) )
        return result
    return wrap

X_op = np.matrix([[0.+0.j, 1.+0.j],
                       [1.+0.j, 0.+0.j]])
Y_op = np.matrix([[ 0.+0.j, -0.-1.j],
               [ 0.+1.j,  0.+0.j]])
Z_op = np.matrix([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]])
I2 = np.matrix([[1.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j]])



def geth_mtx(h):
    for (i,o) in enumerate(h):
        if o == 0:
            O = I2
        if o==1:
            O = X_op
        if o==2:
            O = Y_op
        if o==3:
            O = Z_op
        ###############
        if i==0:
            h_mtx = O 
        else:
            h_mtx = np.kron(O, h_mtx)
    return h_mtx


def getExact(hs, ws):
    ######### exact
    #################
    hs_mtx = []
    for (i,h) in enumerate(hs):
        h_mtx = geth_mtx(h)
        hs_mtx.append(h_mtx)
    
    for (i,h_mtx) in enumerate(hs_mtx):
        if i==0:
            H_mtx = ws[i]*h_mtx
        else:
            H_mtx += ws[i]*h_mtx
    spectrum,v = lg.eigh(H_mtx)
    
    
    E_exact = spectrum[0]
    
    
    return E_exact, hs_mtx, H_mtx, spectrum





# ################################################  ITE
# # @timing
# def getITE(circ_h, circ, delta, machine_precision):
#     psi_t = getState(circ, machine_precision)
#     h_psi_t = getState(circ_h, machine_precision)
    
#     psi_t_delta = psi_t - delta * h_psi_t    
#     norm2 = psi_t_delta.T.conjugate().dot(psi_t_delta)[0,0].real
#     norm = np.sqrt(norm2)
    
#     psi_t_delta = normalize(psi_t_delta)
    
#     return psi_t_delta, norm