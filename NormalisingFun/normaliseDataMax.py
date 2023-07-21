import numpy as np

def normaliseDataMax(data):
    data_mean = np.mean(data)
    data_norm = data - data_mean
    data_max  = np.max(np.abs(data_norm))
    
    if data_max > 1e-8:
        data_norm = data_norm/data_max  
        return data_norm, data_mean, data_max
    else:
        return data

