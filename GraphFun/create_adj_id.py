''' 
    -----------------------------------------------------------------------
        if the data is split, "create_adj_id" renumbers nodes and edges
    ----------------------------------------------------------------------- 
'''

import pandas as pd

def create_adj_id(node_df,edge_df, PRINT): # update the indices of the training subset (node indices, source, targets)
    if PRINT:
        print('\n node_df before update')
        print(node_df)

        print('\n edge_df before update')
        print(edge_df)
        
        
    node_df = node_df.reset_index().reset_index()
    edge_df = pd.merge(edge_df,node_df[['level_0','index']].rename(columns={"level_0":"new_source"}),
                       how='left',left_on='source',right_on='index').drop(columns=['index'])
    
    edge_df = pd.merge(edge_df,node_df[['level_0','index']].rename(columns={"level_0":"new_target"}),
                       how='left',left_on='target',right_on='index').drop(columns=['index'])
    
    edge_df.dropna(inplace=True) # remove rows containing missing values (NaN)
    
    # print here for check
    if PRINT:
        print('\n node_df after update')
        print(node_df)

        print('\n edge_df after update')
        print(edge_df)
        
    # drop extra columns and rename source and targets
    node_df = node_df.drop(columns=['level_0'])
    node_df = node_df.drop(columns=['index'])
    edge_df = edge_df.drop(columns=['source'])
    edge_df = edge_df.drop(columns=['target'])   
    edge_df = edge_df.rename(columns={'new_source' :'source'})
    edge_df = edge_df.rename(columns={'new_target' :'target'}) 
    
    return node_df, edge_df