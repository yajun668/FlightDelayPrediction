# this file contains some functions used for network

#import packages
import time,calendar
from datetime import datetime
from datetime import time as datetime_time
import networkx as nx
import pandas as pd
import numpy as np
import math
from sympy import * #use to calculate a variable value given a function

#this function is to create top N airports matrix regarding #flights between airports
def create_frequency_matrix(df, month, top_airports):
    """
    - df: one of Topa20,Topa30,Topa64,Topa20_delay,Topa20_delay15,Topa30_delay,Topa30_delay15,Topa64_delay,Topa64_delay15
    - top_airports: one of Top20AP,FAA_Large,All.
    """
    df_filtered = df[df['MONTH'] == month]
    frequency_matrix = pd.crosstab(df_filtered['ORIGIN'], df_filtered['DEST'], rownames=['ORIGIN'], colnames=['DEST'], dropna=False)
    frequency_matrix_reindexed = frequency_matrix.reindex(index=top_airports, columns=top_airports, fill_value=0)
    frequency_df = pd.DataFrame(frequency_matrix_reindexed)
    return frequency_df

#this function is to generate a directed graph and network features
def get_graph_features(df,outputname):
  #df: is basically the output-- frequency_df
  #outputname is used to generate output csv file
  column_names = list(df.columns)
  #print(list(column_names))

  #create a directed graph
  G = nx.DiGraph()
  # iterate each pair of nodes
  num_nodes = len(column_names)
  for idx1 in range(num_nodes):
    node1 = column_names[idx1] # node 1 name
    for idx2 in range(num_nodes):
      if idx1==idx2:
        continue
      node2 = column_names[idx2] # node 2 name
      edge_weight = df[column_names[idx1]][idx2]
      if edge_weight>0: #if the number of flights between a pair of airports is greater than 0, then there is an edge;
        #print('node 1-idx, node 2-idx2, and weight are: ', node1,idx1, node2,idx2, edge_weight)
        G.add_edge(node1,node2, weight = edge_weight) # add an edge with nodes and weight
  #print('Graph G is', G)
  #find graph features
  graph_features = {}
  for node in G.nodes():
    graph_features[node] = {}
    graph_features[node]['in_degree']= G.in_degree(node) # node in degree
    graph_features[node]['out_degree']= G.out_degree(node) # node out_degree
    graph_features[node]['weighted_in_degree']= G.in_degree(weight='weight')[node] # weighted in degree
    graph_features[node]['weighted_out_degree']= G.out_degree(weight='weight')[node] # weighted out degree
    graph_features[node]['betweenness_centrality']= nx.betweenness_centrality(G)[node] # betweenness_centrality
    graph_features[node]['closeness_centrality']= nx.closeness_centrality(G)[node] #closeness_centrality
    graph_features[node]['in_degree_centrality']= nx.in_degree_centrality(G)[node] #in_degree_centrality
    graph_features[node]['out_degree_centrality']= nx.out_degree_centrality(G)[node] #out_degree_centrality

  #write graph features into csv file
  new_df = pd.DataFrame(graph_features)
  #print(new_df)
  new_df.to_csv(outputname)
  return new_df

import pandas as pd
from igraph import Graph, plot
import matplotlib.pyplot as plt

#this graph is used to visualize a grap given a adjacency matrix
def visualize_graph(csv_file, output_file):
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file, index_col=0)

    # Create undirected graph
    G = Graph(directed=True)

    # Add nodes to the graph
    nodes = df.index.tolist()
    G.add_vertices(nodes)

    # Add edges to the graph with weights
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            weight = df.iloc[i, j]
            if weight != 0:
                G.add_edge(nodes[i], nodes[j], weight=weight)

    # Calculate the maximum edge weight
    max_edge_weight = max([edge['weight'] for edge in G.es])

    # Define edge widths proportional to edge weight
    edge_widths = [1 + 3 * (edge['weight'] / max_edge_weight) for edge in G.es]

    # Calculate the weighted degree of each node
    weighted_degrees = [sum(G.es.select(_source=node)['weight']) for node in nodes]

    # Normalize the weighted degrees to set node size
    max_weighted_degree = max(weighted_degrees)
    node_sizes = [30 + 100 * (wd / max_weighted_degree) for wd in weighted_degrees]

    # Define a colormap to map node sizes to colors
    colormap = plt.cm.viridis  # You can change the colormap here
    norm = plt.Normalize(min(node_sizes), max(node_sizes))
    colors = [colormap(norm(size)) for size in node_sizes]

    # Define visual style
    visual_style = {
        "bbox": (1500, 1000),  # Adjust the canvas size as needed
        "edge_width": edge_widths
    }
    visual_style["margin"] = 80
    # Visualize the graph with adjusted node size, edge width, and color
    layout = G.layout_kamada_kawai()
    plot(G, output_file, layout=layout, vertex_label=nodes, vertex_size=node_sizes, vertex_color=colors, **visual_style)