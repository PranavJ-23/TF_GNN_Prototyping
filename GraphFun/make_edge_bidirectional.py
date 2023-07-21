import pandas as pd

def make_edge_bidirectional(edge_df): # repeat edges with switch of source and target
    reverse_df = edge_df.rename(columns={'source':'target','target':'source'})
    reverse_df = reverse_df[edge_df.columns]
    reverse_df = pd.concat([edge_df, reverse_df], ignore_index=True, axis=0)
    return reverse_df