#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 12:45:44 2021

@author: jouzdanip
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:14:01 2019

@author: jouzdanip
"""


import os 

import numpy as np

def createPermutationConfigs(numQubits, numFermions):
    '''
    The goal here is to generate the "complete" 
    set of all possible configurations with a given 
    number of fermions and a given number of states.
    Note we are assuming microcanonincal situation, it means 
    in reality not all possible configs are relevant.
    
    Note: Be aware of distinguishability. A prior counting and knowledge of
    the expected number of configs is a good way. Example: numFermion = 3
    numQubits =5 the length of configs_list (see Returns) is 5!/(3! 2!)
    
    
    
    Parameters
    ----------
    numQubits: int
        is the len of the array of the configurations. 
        This array is initialized to 0's. ex. [0,...,0].
        and the lenth of it is numQubits.
        
    numFermions: int
        is the number of 1's in the list. for example if 
        numQubits = 4, and numFermions = 2, [1,0,1,0] is 
        an example of the configurations.
        
    
    
    Returns
    -------
    configs_list : list
        is a list of lists ! it is a list of all possible 'configs'. A 
        config is a list like [1,0,1,0] that has a numFermions 1's and the
        rest of entries are zeros.
 
    '''
    base_dir = os.path.basename(os.path.dirname(__file__))
    file_name = os.path.splitext(os.path.basename(__file__))[0]   
    func_name = createPermutationConfigs.__name__
    print('----> ', base_dir, '--> ', file_name, '--> ', func_name)
    
    
    if numFermions==0:
        return [[0 for q in range(numQubits)]]
    
    def func(i1, numQubits, unusedFermions, array, array_list):
        for i2 in range(i1, numQubits-unusedFermions):
            array[i2]=1
#            print (i1, i2, '\n')
            if unusedFermions!=0:
                
                func(i2+1, numQubits, unusedFermions-1, array, array_list)
#                print(array_list)
            else:
                array_list.append(np.copy(array).tolist())
#                print (array, '\n')
                array[i2]=0
            array[i2]=0
                
        return array_list
    
    
    return func(0, numQubits, numFermions-1, [0 for q in range(numQubits)],
                                            [])

if __name__=='__main__':
    configs_set = createPermutationConfigs(12, 2)      
        
        
        
    
