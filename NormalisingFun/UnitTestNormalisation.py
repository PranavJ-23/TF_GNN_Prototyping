# -*- coding: utf-8 -*-
"""
 Unit Test for data and dataframe normalisation
 
"""

import pandas as pd
import numpy as np
from normaliseData       import normaliseData 
from normaliseDataMax    import normaliseDataMax 
from normalise_dataframe import normalise_dataframe 
import plotly.graph_objects as go
from   plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"


data = np.random.randn(1000)*3 + 2 
norm_data, mean, std = normaliseData(data)
assert abs(np.mean(norm_data))<1e-10,  'mean not zero'
assert abs(np.std(norm_data)-1)<1e-10, 'std not unity'


data = np.random.rand(1000)*3 + 2 
norm_data, mean, std = normaliseData(data)
assert abs(np.mean(norm_data))<1e-10,  'mean not zero'
assert abs(np.std(norm_data)-1)<1e-10, 'std not unity'


n = 1000
float_data =  np.random.randn(n)*3 + 2 
int_data   = np.random.randint(0,2,n,dtype='int32')
bool_data  = np.random.randint(0,2,n,dtype='bool')


NodeDict = {'float_data':  float_data,
            'int_data'  :  int_data,
            'bool_data' :  bool_data}

df = pd.DataFrame(NodeDict)
df_norm, df_mean, df_std = normalise_dataframe(df,'STD')

assert abs(np.mean(df_norm['float_data']))<1e-10,  'mean not zero'
assert abs(np.std(df_norm['float_data'])-1)<1e-10, 'std not unity'
assert abs(np.mean(df_norm['int_data']))<1e-10,  'mean not zero'
assert abs(np.std(df_norm['int_data'])-1)<1e-10, 'std not unity'
assert abs(np.mean(df_norm['bool_data']))>1e-6,  'mean zero for bool data'
assert abs(np.std(df_norm['bool_data'])-1)>1e-6, 'std unity for bool data'


df = pd.DataFrame(NodeDict)
df_norm, df_mean, df_max = normalise_dataframe(df,'MAX')
assert abs(np.mean(df_norm['float_data']))<1e-10,  'mean not zero'
assert abs(np.mean(df_norm['int_data']))<1e-10,  'mean not zero'
assert abs(np.mean(df_norm['bool_data']))>1e-6,  'mean zero for bool data'

assert (abs(np.max(np.abs((df_norm['float_data']))))-1)<1e-10,  'max not unity'
assert (abs(np.max(df_norm['int_data']))-1)<1e-10,  'max not unity'


# fig  = make_subplots(rows=1, cols=2)  # fig = go.Figure()
# fig.add_trace(go.Histogram(
#     x=data,
#     nbinsx=30,
#     marker=dict(color='blue'),  # set the color of the bars
#     opacity=1,  # set the opacity of the bars
#     name='data',
#     histnorm='percent',
# ),row=1,col=1)
# fig.add_trace(go.Histogram(
#     x=norm_data,
#     nbinsx=30,
#     marker=dict(color='red'),  # set the color of the bars
#     opacity=1,  # set the opacity of the bars
#     name='data',
#     histnorm='percent',
# ),row=1,col=2)
# fig.show()