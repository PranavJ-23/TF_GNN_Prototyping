# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 08:27:26 2023

@author: tm16524
"""

def denormaliseData(data,mean,std):
    data = data*std + mean
    return data