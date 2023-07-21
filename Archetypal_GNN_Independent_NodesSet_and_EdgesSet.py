'''
    Archetypal example of GNN Feedforward MLP, including multiple node and edge sets
    
    Nodes have a single feature: node_feature
    Edges have a single feature: edge_feature

    Trying to predict, node outputs z = sum(x_i) for i being all adjacent nodes
    
    hence to work, the graph must retrieve node data from neighbouring nodes and sum them up.
'''
 
import pdb # use pdb.set_trace() for debuging
import tensorflow     as tf
from   tensorflow import keras
from   keras      import layers
import tensorflow_gnn as tfgnn

import plotly.graph_objects as go
from   plotly.subplots  import make_subplots
import plotly.io            as pio
pio.renderers.default = "browser"

import numpy  as np
import pandas as pd

from GraphFun.generate_graph_Econn      import generate_graph_Econn
from GraphFun.plotGraph                 import plotGraph
from GraphFun.make_edge_bidirectional   import make_edge_bidirectional
from NormalisingFun.normalise_dataframe import normalise_dataframe



''' 
    -----------------------------------------------------------------------
        Data Generation Fct.  
        Graph Connectivity + label function = sum of adjacent node features
    ----------------------------------------------------------------------- 
'''

def find_neighbourNodes(Econn,NodeID):
    neighbours_id = np.hstack([Econn[Econn[:,0] == NodeID,1], Econn[Econn[:,1] == NodeID,0]])
    return neighbours_id.tolist()

def find_neighbourEdges(Econn,NodeID):
    edge_id = np.logical_or(Econn[:,0]==NodeID, Econn[:,1]==NodeID)
    edge_id = np.where(edge_id)[0].tolist()
    return edge_id 


def create_df_data(n_nodes): 
    TEST = False
    
    if TEST:  # testing dataset
        n_nodes = 5
        node_feature = np.array([1,2,3,4,5]) # node value = node id
        edge_feature = np.array([1,2,3,4,5]) # edge value = edge id
        
        Econn = np.array([[0,1],
                 [1,2],
                 [0,3],
                 [2,3],
                 [3,4]])
        
        # expected values of node output, using only Node, Edge, or both, up to 1 or 2 edges away
        label_ana_1hop_N   = [7, 6, 9, 13, 9]
        label_ana_1hop_E   = [4, 3, 6, 12, 5]
        label_ana_1hop_NE  = [node_feature + edge_feature for node_feature, edge_feature in zip(label_ana_1hop_N, label_ana_1hop_E)]
        
        label_ana_2hops_N  = [15, 10, 15, 15, 13]
        label_ana_2hops_E  = [15, 10, 15, 15, 12]
        label_ana_2hops_NE = [node_feature + edge_feature for node_feature, edge_feature in zip(label_ana_2hops_N, label_ana_2hops_E)]
    
    
    if not(TEST): # random dataset 
        ne, Econn,_  = generate_graph_Econn(n_nodes,[1, 3])
        node_feature = 1 + 0.5*np.random.uniform(-1, 1, size=[n_nodes,1])
        edge_feature = 1 + 0.5*np.random.uniform(-1, 1, size=[ne,1])
           
        
        
    # calculate the output value z = sum(x_i) 
    # for each node, find the adjacent node up to "n_hops" edges away then sum all the values up
    n_hops = 1                  # only 1 or 2
    node_in_label = True        # include node features in label calcuations 
    edge_in_label = True       # include edge features in label calcuations 
    label  = np.zeros([n_nodes,1])
    
    # -----------------------------
    if n_hops==1: 
        if node_in_label:
            # for each node, find the adjacent nodes then sum all the values in label
            
            adj_list = []
            for i in range(n_nodes):
                label[i] = label[i] + node_feature[i]
                adj_list.append(np.hstack([Econn[Econn[:,0] == i,1], Econn[Econn[:,1] == i,0]]))
                label[i] = label[i] + 1*sum(node_feature[adj_list[i]].flatten())
                
        if edge_in_label:      
            # for each node, find the adjacent edges then substract the sum all the values
            for i in range(n_nodes):
                Edge_ids = np.logical_or(Econn[:,0]==i, Econn[:,1]==i) # edges connected to node i, either as source or target
                label[i] = label[i] + 1*sum(edge_feature[Edge_ids])  
        
    # -----------------------------
    if n_hops==2: 
        if node_in_label: 
            for i in range(n_nodes):
                label[i] = label[i] + node_feature[i]
                neighbours_id_1 = find_neighbourNodes(Econn,i) # find 1st neighbours  (1 edge away)
                adj_list = neighbours_id_1
                for j in range(len(neighbours_id_1)):
                    neighbours_id_2 = find_neighbourNodes(Econn,neighbours_id_1[j]) # find 2nd neighbours (2 edges away)
                    adj_list = adj_list + (neighbours_id_2)
                        
                unique_list = list(set(adj_list))
                label[i] = sum(node_feature[unique_list])
        
        if edge_in_label: 
            for i in range(n_nodes):
                edge_id1        = find_neighbourEdges(Econn,i) # find 1st neighbours edges 
                neighbours_id_1 = find_neighbourNodes(Econn,i) # find 1st neighbours nodes
                
                adj_list = edge_id1
                for j in range(len(neighbours_id_1)):
                    edge_id2 = find_neighbourEdges(Econn,neighbours_id_1[j])     # find 2nd neighbours edges
                    adj_list = adj_list + edge_id2
                    
                unique_list = list(set(adj_list))    
                label[i] = label[i] + sum(edge_feature[unique_list])
        
    # -----------------------------            
    # pdb.set_trace() 
    if TEST:
        if n_hops==1:
            if node_in_label and not(edge_in_label):
                assert all((label.flatten()-np.array(label_ana_1hop_N).flatten())==0), 'incorrect label calculation'
                
            if node_in_label and edge_in_label:    
                assert all((label.flatten()-np.array(label_ana_1hop_NE)).flatten()==0), 'incorrect label calculation'
                
            if not(node_in_label) and edge_in_label:      
                assert all((label.flatten()-np.array(label_ana_1hop_E).flatten())==0), 'incorrect label calculation'
                
        if n_hops==2:
            if node_in_label and not(edge_in_label):
                assert all((label.flatten()-np.array(label_ana_2hops_N).flatten())==0), 'incorrect label calculation'
                
            if node_in_label and edge_in_label:    
                assert all((label.flatten()-np.array(label_ana_2hops_NE).flatten())==0), 'incorrect label calculation'
                
            if not(node_in_label) and edge_in_label:      
                assert all((label.flatten()-np.array(label_ana_2hops_E).flatten()==0)), 'incorrect label calculation'                
               
        print('label calculation correct.')
        
    
    # create node and edge dict then dataframe 
    NodeDict = {}
    NodeDict['ids']           = list(range(0,n_nodes))
    NodeDict['node_feature']  = node_feature.flatten()
    NodeDict['label']         = label.flatten()
    
    # uni-directional edges
    EdgeDict = {'edge_feature': edge_feature.flatten(),
                'source': Econn[:,0].flatten(),
                'target': Econn[:,1].flatten()}
    
    node_df = pd.DataFrame(NodeDict)
    edge_df = pd.DataFrame(EdgeDict)

    return node_df, edge_df



n_nodes = 15

''' Training Dataset '''
# two node datasets and 3 edge datset are created 
# column of boolean are added to identify which set the data belongs to
# edgeset 1  connects element in nodeset 1
# edgeset 2  connects element in nodeset 2
# edgeset 12 connects element between nodesets 1 & 2
#
# Caution Note on Indices:
#    Source and Target indices in an edgeset ALWAYS refer to the nodeset index.
#    Even if the data is concatenated into node_df_train, edge_df_train
#    
# Example:
#   nodeset1_id = [0,1,2,3,4]
#   nodeset2_id = [0,1,2]
#   node_df_if  = [0,1,2,3,4,   5,6,7]
# 
#   edgeset_1: [source=0, target=1] connects nodes in nodeset 1
#   edgeset_2: [source=0, target=1] connects nodes in nodeset 2
# 
# i.e. there is no need to add an offset to the source and target in edgeset_2
#    

node_df_train1, edge_df_train1 = create_df_data(n_nodes)            # training set1
node_df_train2, edge_df_train2 = create_df_data(n_nodes)            # training set2

node_df_train1['set']=np.ones(n_nodes,dtype='int')*1
node_df_train2['set']=np.ones(n_nodes,dtype='int')*2
edge_df_train1['set']=np.ones(len(edge_df_train1),dtype='int')*1
edge_df_train2['set']=np.ones(len(edge_df_train2),dtype='bool')*2

edge_df_train1['source_nodeset']=np.ones(len(edge_df_train1),dtype='int')*1
edge_df_train1['target_nodeset']=np.ones(len(edge_df_train1),dtype='int')*1
edge_df_train2['source_nodeset']=np.ones(len(edge_df_train2),dtype='int')*2
edge_df_train2['target_nodeset']=np.ones(len(edge_df_train2),dtype='int')*2

edge_df_train12={'edge_feature':[0,0] ,
                'source':[0,1],
                'target':[0,1],
                'set':[12,12],
                'source_nodeset':[1,1],
                'target_nodeset':[2,2]}

edge_df_train12 = pd.DataFrame(edge_df_train12)
node_df_train2['label']  = node_df_train2['label'].values + 1
node_df_train = pd.concat([node_df_train1,node_df_train2],axis=0)
edge_df_train = pd.concat([edge_df_train1,edge_df_train2],axis=0)
edge_df_train = pd.concat([edge_df_train,edge_df_train12],axis=0)


''' Validation Dataset '''
node_df_val1, edge_df_val1 = create_df_data(n_nodes)    # valing set1
node_df_val2, edge_df_val2 = create_df_data(n_nodes)    # valing set2

node_df_val1['set']=np.ones(n_nodes,dtype='int')*1
node_df_val2['set']=np.ones(n_nodes,dtype='int')*2
edge_df_val1['set']=np.ones(len(edge_df_val1),dtype='int')*1
edge_df_val2['set']=np.ones(len(edge_df_val2),dtype='bool')*2

edge_df_val1['source_nodeset']=np.ones(len(edge_df_val1),dtype='int')*1
edge_df_val1['target_nodeset']=np.ones(len(edge_df_val1),dtype='int')*1
edge_df_val2['source_nodeset']=np.ones(len(edge_df_val2),dtype='int')*2
edge_df_val2['target_nodeset']=np.ones(len(edge_df_val2),dtype='int')*2


edge_df_val12={'edge_feature':[0,0] ,
                'source':[0,1],
                'target':[0,1],
                'set':[12,12],
                'source_nodeset':[1,1],
                'target_nodeset':[2,2]}

edge_df_val12 = pd.DataFrame(edge_df_val12)
node_df_val2['label']  = node_df_val2['label'].values + 1
node_df_val = pd.concat([node_df_val1,node_df_val2],axis=0)
edge_df_val = pd.concat([edge_df_val1,edge_df_val2],axis=0)
edge_df_val = pd.concat([edge_df_val,edge_df_val12],axis=0)



''' Test Dataset '''
node_df_test1, edge_df_test1 = create_df_data(n_nodes)    # testing set1
node_df_test2, edge_df_test2 = create_df_data(n_nodes)    # testing set2

node_df_test1['set']=np.ones(n_nodes,dtype='int')*1
node_df_test2['set']=np.ones(n_nodes,dtype='int')*2
edge_df_test1['set']=np.ones(len(edge_df_test1),dtype='int')*1
edge_df_test2['set']=np.ones(len(edge_df_test2),dtype='bool')*2

edge_df_test1['source_nodeset']=np.ones(len(edge_df_test1),dtype='int')*1
edge_df_test1['target_nodeset']=np.ones(len(edge_df_test1),dtype='int')*1
edge_df_test2['source_nodeset']=np.ones(len(edge_df_test2),dtype='int')*2
edge_df_test2['target_nodeset']=np.ones(len(edge_df_test2),dtype='int')*2


edge_df_test12={'edge_feature':[0,0] ,
                'source':[0,1],
                'target':[0,1],
                'set':[12,12],
                'source_nodeset':[1,1],
                'target_nodeset':[2,2]}

edge_df_test12 = pd.DataFrame(edge_df_test12)

node_df_test2['label']  = node_df_test2['label'].values + 1
node_df_test = pd.concat([node_df_test1,node_df_test2],axis=0)
edge_df_test = pd.concat([edge_df_test1,edge_df_test2],axis=0)
edge_df_test = pd.concat([edge_df_test,edge_df_test12],axis=0)




''' 
    -----------------------------------------------------------------------
        Dataset  Normalisation
    ----------------------------------------------------------------------- 
'''
node_df_train = node_df_train.reset_index(drop=True)
edge_df_train = edge_df_train.reset_index(drop=True)
node_df_val = node_df_val.reset_index(drop=True)
edge_df_val = edge_df_val.reset_index(drop=True)
node_df_test = node_df_test.reset_index(drop=True)
edge_df_test = edge_df_test.reset_index(drop=True)


NormTYPE = 'STD'
node_df_train,  node_df_train_mean, node_df_train_std   = normalise_dataframe(node_df_train, NormTYPE)
edge_df_train,  edge_df_train_mean, edge_df_train_std   = normalise_dataframe(edge_df_train, NormTYPE)

# different from conventional database, here the normalisation is based on the training and
# applied equally to the other graphs to avoid diffferent normalisations
for iloc in range(len(node_df_val.keys())): 
    my_key = node_df_train.keys()[iloc]
    if node_df_val[my_key].dtype != 'bool' and my_key!='set':
        node_df_val.iloc[:,iloc]  = (node_df_val.iloc[:,iloc]-node_df_train_mean[iloc])/node_df_train_std[iloc]
        node_df_test.iloc[:,iloc] = (node_df_test.iloc[:,iloc]-node_df_train_mean[iloc])/node_df_train_std[iloc]
    
for iloc in range(len(edge_df_train.keys())):
    my_key = edge_df_train.keys()[iloc]
    if edge_df_train[my_key].dtype != 'bool' and my_key!='source' and my_key!='target' and my_key!='set':    
        edge_df_val.iloc[:,iloc]  = (edge_df_val.iloc[:,iloc]-edge_df_train_mean[iloc])/edge_df_train_std[iloc]
        edge_df_test.iloc[:,iloc] = (edge_df_test.iloc[:,iloc]-edge_df_train_mean[iloc])/edge_df_train_std[iloc]
    
edge_df_train = make_edge_bidirectional(edge_df_train)
edge_df_val   = make_edge_bidirectional(edge_df_val)
edge_df_test  = make_edge_bidirectional(edge_df_test)



if len(node_df_train)<=100:
    plotGraph(node_df_train, edge_df_train)


print('---------------------')
print(np.mean(node_df_train['node_feature'].values))
print(np.std(node_df_train['node_feature'].values))

print(np.mean(node_df_val['node_feature'].values))
print(np.std(node_df_val['node_feature'].values))

print(np.mean(node_df_test['node_feature'].values))
print(np.std(node_df_test['node_feature'].values))

print('---------------------')
print(np.mean(edge_df_train['edge_feature'].values))
print(np.std(edge_df_train['edge_feature'].values))

print(np.mean(edge_df_val['edge_feature'].values))
print(np.std(edge_df_val['edge_feature'].values))

print(np.mean(edge_df_test['edge_feature'].values))
print(np.std(edge_df_test['edge_feature'].values))



''' 
    -----------------------------------------------------------------------
        Create the graph tensors
        
        x_norm are the node features we use as inputs 
        z_norm are the node ouptut  we are trying to predict
        note that z_norm will be removed from the features in the "node_batch_merge" function called later
        and set as the label we are trying to match
        
        y_norm are the edge features, which we remove in this example
        
        only type available to date are  dtype='float32' and  dtype='int32'
    ----------------------------------------------------------------------- 
'''
def create_graph_tensor(node_df, edge_df):
    
    id_ns1  = node_df['set']==1
    id_ns2  = node_df['set']==2
    id_es1  = edge_df['set']==1
    id_es2  = edge_df['set']==2   
    id_es12 = edge_df['set']==12

    # pdb.set_trace()
    
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets = {
            "node_set1": tfgnn.NodeSet.from_fields(
                sizes = [sum(id_ns1)],
                features ={
                            'node_feature': np.array(node_df['node_feature'][id_ns1], dtype='float32').reshape(sum(id_ns1),1), # input feature
                            'label':        np.array(node_df['label'][id_ns1], dtype='float32').reshape(sum(id_ns1),1),        # label
                          }),
        
            "node_set2": tfgnn.NodeSet.from_fields(
                sizes = [sum(id_ns2)],
                features ={
                            'node_feature': np.array(node_df['node_feature'][id_ns2], dtype='float32').reshape(sum(id_ns2),1), # input feature
                            'label':        np.array(node_df['label'][id_ns2], dtype='float32').reshape(sum(id_ns2),1),        # label
                          })},
        
        edge_sets ={
            "edge_set1": tfgnn.EdgeSet.from_fields(
                sizes = [sum(id_es1)],
                features = {
                    'edge_feature': np.array(edge_df['edge_feature'][id_es1],  dtype='float32').reshape(sum(id_es1),1),        # inputs
                    },
                adjacency = tfgnn.Adjacency.from_indices(
                    source = ("node_set1", np.array(edge_df['source'][id_es1], dtype='int32')),
                    target = ("node_set1", np.array(edge_df['target'][id_es1], dtype='int32')))),
            
            "edge_set2": tfgnn.EdgeSet.from_fields(
                sizes = [sum(id_es2)],
                features = {
                    'edge_feature': np.array(edge_df['edge_feature'][id_es2],  dtype='float32').reshape(sum(id_es2),1),        # inputs
                    },
                adjacency = tfgnn.Adjacency.from_indices(
                    source = ("node_set2", np.array(edge_df['source'][id_es2], dtype='int32')),
                    target = ("node_set2", np.array(edge_df['target'][id_es2], dtype='int32'))
                    )),
            
            "edge_set12": tfgnn.EdgeSet.from_fields(
                sizes = [sum(id_es12)],
                features = {
                    'edge_feature': np.array(edge_df['edge_feature'][id_es12],  dtype='float32').reshape(sum(id_es12),1),        # inputs
                    },
                adjacency = tfgnn.Adjacency.from_indices(
                    source = ("node_set1", np.array(edge_df['source'][id_es12], dtype='int32')),
                    target = ("node_set2", np.array(edge_df['target'][id_es12], dtype='int32'))
                    ))
      })
    return graph_tensor

train_tensor = create_graph_tensor(node_df_train, edge_df_train)
val_tensor   = create_graph_tensor(node_df_val,   edge_df_val)
test_tensor  = create_graph_tensor(node_df_test,  edge_df_test)


'''
    -----------------------------------------------------------------------
        TensorFlow Dataset from graph    
        the order of operation is
        
        1. We create our dataset from the graph tensor.
        2. We split our dataset in batches (read up on batch sizes).
        3. In the map function, we merge those batches back into one graph.
        4. We split/drop the features as needed.
    -----------------------------------------------------------------------
'''

def node_batch_merge(graph):
    graph              = graph.merge_batch_to_components()
    node_features_set1 = graph.node_sets['node_set1'].get_features_dict()  # returns a dict of tensors
    edge_features_set1 = graph.edge_sets['edge_set1'].get_features_dict()  # returns a dict of tensors
    
    node_features_set2 = graph.node_sets['node_set2'].get_features_dict()  # returns a dict of tensors
    edge_features_set2 = graph.edge_sets['edge_set2'].get_features_dict()  # returns a dict of tensors

    # label = node_features_set1.pop('label')
    # _ = node_features_set2.pop('label')
    
    # label = node_features_set2.pop('label')
    # _ = node_features_set1.pop('label')
    
    label_1 = node_features_set1.pop('label')
    label_2 = node_features_set2.pop('label')
    label = tf.concat([label_1, label_2], axis=0) 

    new_graph = graph.replace_features(
        node_sets={'node_set1':node_features_set1,
                   'node_set2':node_features_set2
                   },
        edge_sets={'edge_set1':edge_features_set1,
                   'edge_set2':edge_features_set2}
        )
    
    return new_graph, label

batch_size = 32
def create_dataset(graph,function):
    dataset = tf.data.Dataset.from_tensors(graph)   # 1
    dataset = dataset.batch(batch_size)                     # 2
    return dataset.map(function)                    # 3


#Node Datasets
train_node_dataset = create_dataset(train_tensor,node_batch_merge)
val_node_dataset   = create_dataset(val_tensor,node_batch_merge)
test_node_dataset  = create_dataset(test_tensor,node_batch_merge)



'''
    -----------------------------------------------------------------------
        Example of visualisation of dataset  
        similar to that of a graph, since the dataset contains the graph 
        the label is stored as the output to be matches somehow
        

        dataset      = tf.data.Dataset.from_tensors(full_tensor) 
        element_spec = dataset.element_spec

        # Iterate through the elements using a loop
        for element in dataset:
            print(element)
            print(element.edge_sets['element'].features)
            
            
        # Iterate through the elements using a loop
        for node in dataset:
            print(node)
            print(node.node_sets['node'].features)
            
            
        # Iterate through the elements using a loop
        for element in full_node_dataset: # element in this case is a tuple
            myGraphTensor = element[0]
            mylabel = element[1] # == U_norm
            
            print('\n my edge sets features')
            print(myGraphTensor.edge_sets['element'].features)
            print('\n my node sets features')                   
            print(myGraphTensor.node_sets['node'].features)         # does not containt U_norm anymore! 
            print('\n my label')
            print(mylabel) # == U_norm
            
    -----------------------------------------------------------------------
'''


 



'''
    -----------------------------------------------------------------------
        Model 
    -----------------------------------------------------------------------
'''
graph_spec  = train_node_dataset.element_spec[0]
input_graph = tf.keras.layers.Input(type_spec=graph_spec) # KerasTensor



'''
    -----------------------------------------------------------------------
        Feature Embeddings: add layers between normalised inputs and GNN input
    -----------------------------------------------------------------------
'''
n_mapping_neurons = 0
def set_initial_node_state(node_set, node_set_name):
    if n_mapping_neurons == 0:
        features = node_set['node_feature']  # works (no mapping)
    else:     
        features = tf.keras.layers.Dense(n_mapping_neurons,activation="relu")(node_set['node_feature']) # works (1 layer mapping) 
        # features = tf.keras.layers.Dense(8,activation="relu")(tf.keras.layers.Dense(8,activation="relu")(node_set['node_feature'])) # works (2 layer mapping)
    return features
    # Or we can also have multiple features
    # original_features = node_set['x_norm']
    # embeded_features  = tf.keras.layers.Dense(8,activation="relu")(node_set['x_norm'])   
    # return tf.keras.layers.Concatenate()([original_features, embeded_features])



def set_initial_edge_state(edge_set, edge_set_name):
    if n_mapping_neurons == 0:
        features = edge_set['edge_feature'] # no mapping
    else:
        features = tf.keras.layers.Dense(8,activation="relu")(edge_set['edge_feature'])
        
    return features


graph = tfgnn.keras.layers.MapFeatures(
    node_sets_fn=set_initial_node_state,
    edge_sets_fn=set_initial_edge_state
)(input_graph)  # KerasTensor



'''
    -----------------------------------------------------------------------
        custom dense layers
    -----------------------------------------------------------------------
'''
def dense_layer(units=1, l2_reg=0.1, activation='relu'):
    regularizer = tf.keras.regularizers.l2(l2_reg)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units,
                              kernel_regularizer=regularizer,
                              bias_regularizer=regularizer,
                              activation=activation)])


def n_dense_layer(n_layers=1, units=8,l2_reg=0.01, dropout=0, activation='relu'):
    regularizer = tf.keras.regularizers.l2(l2_reg)
    model = tf.keras.Sequential() 
    for i in range(n_layers):
        model.add(layers.Dense(units, activation=activation, kernel_regularizer=regularizer ))
        
        if dropout>0:
            model.add(layers.Dropout(dropout))
    
    return model
   


'''
    GRPAH UPDATE
    https://towardsdatascience.com/tensorflow-gnn-an-end-to-end-guide-for-graph-neural-networks-a66bfd237c8c
    https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/gnn_modeling.md
    
    The code below can seem a little confusing because of how TensorFlow stacking works. 
    Remember that the (graph) labeled ‘#start here’ at the end of the ‘GraphUpdate’ function is 
    really the input for the code that comes before it. At first, this (graph) equals the initialized
    features we mapped previously. The input gets fed into the ‘GraphUpdate’ function becoming the new (graph).
    With each ‘graph_updates’ loop, the previous ‘GraphUpdate’ becomes the input for the new ‘GraphUpdate’
    along with a dense layer specified with the ‘NextStateFromConcat’ function. This diagram should help explain:
    
    INPUT ---->  Graph update 1                             ----> Graph update i
    INPUT ---->  Convolution (12) ----> 8 Dense Layer       ----> ...
    
    In this example a single graph update consist of a convolutional layer and a dense layer.
    The convolutional layer when applied at a node, regroup information of that nodes with the information of
    neighbour edges and nodes.
    
    The ‘GraphUpdate’ function simply updates the specified states (node, edge, or context) and adds a next state layer.
    In this case, we are only updating the node states with ‘NodeSetUpdate’ but we will explore an edge-centric approach
    when we work on our edge model. With this node update, we are applying a convolutional layer along the edges, 
    allowing for information to feed to the node from neighboring nodes and edges.
    
    The number of graph updates is a tunable parameter, with each update allowing for information to travel 
    from further nodes. For example, the three updates specified in our case allow for information to travel 
    from up to three nodes away. After our graph updates, the final node state becomes the input for our prediction
    head labeled ‘logits’. Because we are predicting 12 different conferences, we have a dense layer of 12 units with
    a softmax activation. Now we can compile the model.
    
    
    The tfgnn.keras.layers.SimpleConv used applies the passed-in transformation Dense(..., "relu") 
    on the concatenated source and target node states of each edge and then pools the result for each receiver node,
    with a user-specified pooling method like "sum" or "mean"
    
    In the case of multiple update (say for different sets), the layer
    tfgnn.keras.layers.NextStateFromConcat regroup this information
    
    The library-provided tfgnn.keras.layers.NodeSetUpdate combines two kinds of pieces that do actual computations:
    Convolutions per edge set towards the updated node set.
    One next-state layer to compute a new node state from the old node state and from the results of the convolutions.
    
    At each node, a convolution computes ONE result for each EDGE SET (aggregated across the variable number of incident edges),
    but computing the new node state from the fixed number of edge sets is left to the next-state layer.
    The example above shows the library-provided NextStateFromConcat, which concatenates all inputs and sends 
    them through a user-supplied projection for computing the new node state.
     
'''
n_layers            = 1
l2_reg              = 1e-8
dropout             = 0 # between 0 and 1
activation          = 'relu'
n_neurons_per_layer = 1
n_concat_neurons    = 1


graph_updates = 1
for i in range(graph_updates):
    graph = tfgnn.keras.layers.GraphUpdate( 
        # edge update 
        # for each edge, concatenate Target and Source node features
        # then pass it through a dense layer, which has n_concat_neurons output
        #
        # node update
        # for each node, if connected to multiple edges, the results of each updated edge features are pooled (summed)
        # then concatenated with the current node feature and the results is pass through another concat layer 
        #
        edge_sets = {
                    'edge_set1': 
                    tfgnn.keras.layers.EdgeSetUpdate(
                                            next_state=tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(n_concat_neurons)),
                                            edge_input_feature      = tfgnn.HIDDEN_STATE,
                                            node_input_tags         = (tfgnn.SOURCE, tfgnn.TARGET),
                                            node_input_feature      = tfgnn.HIDDEN_STATE,
                                            context_input_feature   = None      ),
                    
                    
                    'edge_set2': 
                    tfgnn.keras.layers.EdgeSetUpdate(
                                            next_state=tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(n_concat_neurons)),
                                            edge_input_feature      = tfgnn.HIDDEN_STATE,
                                            node_input_tags         = (tfgnn.SOURCE, tfgnn.TARGET),
                                            node_input_feature      = tfgnn.HIDDEN_STATE,
                                            context_input_feature   = None      )
                    },
                        
       node_sets={
                   "node_set1":
                  tfgnn.keras.layers.NodeSetUpdate(  
                        {"edge_set1": tfgnn.keras.layers.Pool(tfgnn.SOURCE, "sum") },
                        tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(n_concat_neurons))),

                   "node_set2":
                  tfgnn.keras.layers.NodeSetUpdate(  
                        {"edge_set2": tfgnn.keras.layers.Pool(tfgnn.SOURCE, "sum") },
                        tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(n_concat_neurons)))
                  
                  })(graph) #start here

    # output = tf.keras.layers.Dense(1,activation='linear')(graph.node_sets["node_set1"][tfgnn.HIDDEN_STATE])
    # output = tf.keras.layers.Dense(1,activation='linear')(graph.node_sets["node_set2"][tfgnn.HIDDEN_STATE])
    
    output = tf.keras.layers.Dense(1,activation='linear')\
                ( tf.concat( 
                              [graph.node_sets["node_set1"][tfgnn.HIDDEN_STATE] ,
                              graph.node_sets["node_set2"][tfgnn.HIDDEN_STATE] ], axis=0 ))

    
node_model = tf.keras.Model(input_graph, output)


##### compile
learning_rate = 0.02
node_model.compile(
    tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss = tf.keras.losses.MeanSquaredError()
)

node_model.summary()


es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',mode='min',verbose=1,
        patience=10,restore_best_weights=True)


history = node_model.fit(train_node_dataset.repeat(),
                           validation_data=val_node_dataset,
                           steps_per_epoch=10,
                           epochs=125,
                           callbacks=[es])




n_dataset = n_nodes
print('''
      # graph_updates:  %i
      # n_layers:       %i
      # n_mapping_neurons: %i 
      # n_concat_neurons:  %i
      # n_dataset:      %i
      learning_rate:    %f
      training loss:    %f
      validation loss:  %f ''' 
      %(graph_updates, n_layers, n_mapping_neurons, n_concat_neurons, n_dataset,
        learning_rate, history.history['loss'][-1], history.history['val_loss'][-1]))



loss = node_model.evaluate(train_node_dataset)
loss = node_model.evaluate(val_node_dataset)

import matplotlib.pyplot as plt
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  # plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  
plot_loss(history)

stop

# True Vs Prediction plot
y    = np.array(node_df_train['label'])
yhat = node_model.predict(train_node_dataset).flatten()
Err  = abs((y-yhat))

y_val    = np.array(node_df_val['label'])
yhat_val = node_model.predict(val_node_dataset).flatten()
Err      = abs((y_val-yhat_val))

fig = go.Figure()

fig.add_trace(go.Scatter( x=[-15,15], y=[-15,15], mode = 'lines', line=dict(color='black')))

fig.add_trace(go.Scatter(x=y.flatten(),
                         y=yhat.flatten(),
                         mode = 'markers',
                         line    = dict(color='blue'),
                         marker  = dict(color='blue'),
                         name= 'label ',
                         ))

fig.add_trace(go.Scatter(x=y_val.flatten(),
                         y=yhat_val.flatten(),
                         mode = 'markers',
                         line    = dict(color='red'),
                         marker  = dict(color='red'),
                         name= 'label ',
                         ))
fig.update_xaxes( title_text="True label")
fig.update_yaxes( title_text="Predicted label")


fig.show()








##### retrieving weights

node_model.get_weights()

weights = node_model.weights
for weight in weights:
    weight_values = weight.numpy().flatten()
    layer_name    = weight.name.split('/')[0]  # Extract the layer name from the weight name
    print(f"Weight name: {weight.name}, Layer name: {layer_name}, Weight Values: {weight_values}")
   
    
# layer order 
# [0] input
# [1] map feature
# [2] graph update (include mnessage + nextstatefromConcat)
# [5] output layer

# input_129
# map_features_64
# graph_update_79
# input.node_sets_71 [?]
# input._get_features_ref_136 [?]
# dense_244

for i in range(len(node_model.layers)):
    target_layer = node_model.layers[i]   
    print(target_layer.name)
    print(target_layer.get_weights())
















