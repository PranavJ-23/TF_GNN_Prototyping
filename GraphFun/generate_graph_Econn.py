
import numpy as np
import pdb # use pdb.set_trace() for debuging


def generate_graph_Econn (n_nodes, n_edges_perNode):
    # INPUTS
    # n_nodes [int]
    # n_edges_perNode = tupple or list [1, 5] # min and max number of edge for each node
    Econn = []
    number_of_edges = np.zeros(n_nodes,dtype=int)
    # for each node ii, assign a random number of edges, randomly pick nodes
    for ii in range(n_nodes):
        
        # check how many edge the current node has
        if ii>0:
            id_linkedNodes      = np.where( np.logical_or(Econn[:,0]==ii , Econn[:,1]==ii) )[0]
            number_of_edges[ii] = len(id_linkedNodes)   
            id_linkedNodes      = np.unique(Econn[id_linkedNodes,:].flatten()).tolist()
        else:
            number_of_edges[ii] = 0
            id_linkedNodes      = []
        # if lower than the desired number of edges, generate a new unique edge
        n_edge = np.random.randint(low=n_edges_perNode[0], high=n_edges_perNode[1]+1, size=1)
        
        while number_of_edges[ii] < n_edge:
            avoid_list      = list(set([ii] + id_linkedNodes))
            random_node_id  = np.random.choice([x for x in range(n_nodes) if x not in avoid_list], size=1) # add one edge at a time and check if unique
            
            if len(Econn)==0:
                Econn = np.array([ [ii, random_node_id.tolist()[0] ]])
            else:
                Econn = np.concatenate((Econn, [ [ii, random_node_id.tolist()[0] ]]),axis=0)
            
            Econn_temp  = np.concatenate((Econn, np.flip(Econn,axis=1)),axis=0)
            unique_rows = np.unique(Econn_temp, axis=0)
            is_unique   = (unique_rows.shape[0] == Econn_temp.shape[0])
            
            if is_unique:
                id_linkedNodes      = np.where( np.logical_or(Econn[:,0]==ii , Econn[:,1]==ii) )[0]    
                number_of_edges[ii] = len(id_linkedNodes)
                id_linkedNodes      = np.unique(Econn[id_linkedNodes,:].flatten()).tolist()    
            else:
                Econn = Econn[:-1,:]
        
    ne = Econn.shape[0]
    
    return ne, Econn, number_of_edges