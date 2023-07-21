import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def generate_pastel_color():
    # Generate random HSL values
    hue = random.randint(0, 360)
    saturation = random.uniform(0.3, 0.7)
    lightness = random.uniform(0.6, 0.9)

    # Convert HSL to RGB
    h = hue / 360
    s = saturation
    l = lightness

    if s == 0:
        r = g = b = l
    else:
        def hue2rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p

        if l < 0.5:
            q = l * (1 + s)
        else:
            q = l + s - l * s
        p = 2 * l - q
        r = hue2rgb(p, q, h + 1/3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1/3)

    # Scale RGB values to [0, 255]
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)

    return (r, g, b)



def plotGraph(node_df, edge_df):
    node_df = node_df.reset_index(drop=True)
    edge_df = edge_df.reset_index(drop=True)
    
    ##########################################################################
    ##### plot using the local indices of each subgraphs
    ###########################################################################
    # Create empty graph
    G = nx.Graph()
    
    # identify the total number of nodeset and edgesets
    nodeset_ids     = np.unique(node_df['set'])
    edgeset_ids     = np.unique(edge_df['set'])
    n_nodes_perset  = np.zeros(len(nodeset_ids))
    legend_list     = []
    node_labels     = node_df['ids'].values

    # add the nodes belonging to each nodeset
    for i in range(len(nodeset_ids)):
        n_nodes_perset[i]  = sum(node_df['set'] == nodeset_ids[i])
        nodeset_i = node_df[node_df['set'] == nodeset_ids[i]].copy()
        G.add_nodes_from(nodeset_i.index, nodeset='Set'+str(nodeset_ids[i]))
        legend_list = legend_list + ['Node Set'+str(nodeset_ids[i])]
        
    # add the edges belonging to each nodeset
    for i in range(len(edgeset_ids)):
        edgeset_i = edge_df[edge_df['set'] == edgeset_ids[i]].copy() 
        # add the offset for the edges targeting nodeset different from 1
        # implies nodes set start at 1 and increase monotically
        source_nodeset = edgeset_i['source_nodeset'].values[0]
        target_nodeset = edgeset_i['target_nodeset'].values[0]
        source_offset = sum(n_nodes_perset[0:source_nodeset-1])
        target_offset = sum(n_nodes_perset[0:target_nodeset-1])   
        edgeset_i['source'] = edgeset_i['source'].values + source_offset   
        edgeset_i['target'] = edgeset_i['target'].values + target_offset
            
        G.add_edges_from(edgeset_i[['source','target']].values, edgeset='Set'+str(edgeset_ids[i]))      
        legend_list = legend_list + ['Edge Set'+str(edgeset_ids[i])]
        

    # Specify the positions of the nodes
    pos = nx.spring_layout(G)
     
    # plot the nodes belonging to each nodeset
    for i in range(len(nodeset_ids)):
        color     = generate_pastel_color()
        red = color[0]
        green = color[1]
        blue = color[2]
        color = '#%02x%02x%02x' % (red, green, blue)

        # color     = f'#{np.random.randint(0, 0xFFFFFF):06x}'
        nodeset_i = node_df[node_df['set'] == nodeset_ids[i]]
        nx.draw_networkx_nodes(G, pos, nodeset_i.index, node_color=color, node_size=100)
    
    # plot the edges belonging to each edgeset
    for i in range(len(edgeset_ids)):
        color     = generate_pastel_color()
        red = color[0]
        green = color[1]
        blue = color[2]
        color = '#%02x%02x%02x' % (red, green, blue)
        # color     = f'#{np.random.randint(0, 0xFFFFFF):06x}'
        edgeset_i = edge_df[edge_df['set'] == edgeset_ids[i]] 
        
        # add the offset for the edges targeting nodeset different from 1
        # implies nodes set start at 1 and increase monotically
        source_nodeset = edgeset_i['source_nodeset'].values[0]
        target_nodeset = edgeset_i['target_nodeset'].values[0]
        source_offset = sum(n_nodes_perset[0:source_nodeset-1])
        target_offset = sum(n_nodes_perset[0:target_nodeset-1])   
        edgeset_i['source'] = edgeset_i['source'].values + source_offset   
        edgeset_i['target'] = edgeset_i['target'].values + target_offset
          
        nx.draw_networkx_edges(G, pos, edgelist=edgeset_i[['source','target']].values, edge_color=color, width=2)
        

    # nx.draw_networkx_labels(G, pos)
    
    # global_labels = pd.concat([df_id_nodeSet1['global_Ids'] , df_id_nodeSet2['global_Ids']])
    # labels    = {node: value for node, value in zip(G.nodes(), global_labels.values)}
    # label_pos = {node: (pos[node][0]+0.05, pos[node][1] - 0.075) for node in G.nodes()}
    # nx.draw_networkx_labels(G, label_pos, labels, font_color='purple', font_size=10, font_weight='bold')
    
    # Add labels to the nodes
    nodes = list(node_df.index)
    labels_list = node_labels.tolist()
    for node, label in zip(nodes, labels_list):
        x, y = pos[node]
        plt.text(x, y, label, color='black', ha='center', va='center')


    # nx.draw_networkx_labels(G, pos, labels=node_labels.tolist(), font_color='black')
    # nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black', font_size=10, font_weight='bold')
    
    plt.axis('off')
    plt.title('Graph with Multiple Edge Sets and Node Sets')
    plt.legend(legend_list)
    plt.show()
        
    

# original function
'''
def plotGraph(node_df,edge_df):
    node_df = node_df.reset_index(drop=True)
    edge_df = edge_df.reset_index(drop=True)
    
    G = nx.Graph()
    for i in range(len(node_df)):
        G.add_node(i)
       
    if any(edge_df.columns == 'source'):
        for i in range(len(edge_df)):
            G.add_edge(edge_df['source'][i], edge_df['target'][i])
            
    elif any(edge_df.columns == 'source_id'):
        for i in range(len(edge_df)):
            G.add_edge(edge_df['source_id'][i], edge_df['target_id'][i])
              
    # edge_numbering = {edge: i+1 for i, edge in enumerate(G.edges())}        
    
    nx.draw_networkx(G,node_size=100)
    plt.show()
'''