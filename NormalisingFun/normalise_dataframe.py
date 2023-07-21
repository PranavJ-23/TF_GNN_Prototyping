''' 
    -----------------------------------------------------------------------
        normalise the data in dataframes (between 0-1)
    ----------------------------------------------------------------------- 
'''

import numpy as np
from NormalisingFun.normaliseData    import normaliseData 
from NormalisingFun.normaliseDataMax import normaliseDataMax 
import pdb # use pdb.set_trace() for debuging

def normalise_dataframe(df,TYPE):
    df_norm = df.copy()
    keys    = df.keys()
    nk      = len(keys)
    df_mean = np.nan*np.zeros(nk)
    df_std  = np.nan*np.zeros(nk)
    df_max  = np.nan*np.zeros(nk)
    
    
    keywords = ['source', 'target', 'set', 'source_nodeset' , 'target_nodeset', 'ids']  # keys that should not be normalised
        
    if TYPE == 'STD':
        for i in range(nk):
            if df[keys[i]].dtype!='bool'  and not(keys[i] in keywords):
                df_norm[keys[i]], df_mean[i], df_std[i] = normaliseData(np.array(df.iloc[:,i].values))            
        return df_norm, df_mean, df_std, 
    
    elif TYPE == 'MAX':
        for i in range(nk):
            if df[keys[i]].dtype!='bool'  and not(keys[i] in keywords):
                df_norm[keys[i]], df_mean[i], df_max[i],  = normaliseDataMax(np.array(df.iloc[:,i].values))     
        return df_norm, df_mean, df_max
    
    else:
        raise ValueError("Input string is not the expected value: use either STD or MAX")
    