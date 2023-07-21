# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 08:26:30 2023

@author: tm16524
"""

import numpy as np

def normaliseData(data):
    mean = np.mean(data)
    std  = np.std(data)
    if std > 1e-8:
        data = (data - mean)/std      
        assert abs(np.mean(data))<1e-13,  'mean not normalised'
        assert abs(np.std(data)-1)<1e-13, 'std not normalised'
    
    return data, mean, std