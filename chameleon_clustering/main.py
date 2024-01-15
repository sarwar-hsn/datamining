import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import matplotlib.pyplot as plt
import math
import metis
import itertools

#Variables to control flow
k = 2 # Number of neighbours in KNN-Graph
selected_columns = [0, 1, 2, 3] #Select which parameters of the dataset you want
max_rows = 200 #maximum number of rows from dataset
distance_algorithm = 'euclidean' #euclidean or manhattan


#Initializing dataset
file_path = 'worms_64d.txt'
worms_data = np.loadtxt(file_path,max_rows=max_rows)


# Extract the columns corresponding to the selected indices
selected_data = worms_data[:, selected_columns]

# #Creating similarity matrix using Euclidean distance. This part is only for understanding data
# similarity_matrix = np.zeros((len(worms_data), len(worms_data)))
# for i in range(len(worms_data)):
#     for j in range(len(worms_data)):
#         similarity_matrix[i][j] = np.linalg.norm(worms_data[i] - worms_data[j]) #euclidean

# # Convert to a DataFrame for better readability
# similarity_matrix_df = pd.DataFrame(similarity_matrix, 
#                                     columns=[f"Worm {i+1}" for i in range(len(worms_data))],
#                                     index=[f"Worm {i+1}" for i in range(len(worms_data))])

# # Display the similarity matrix
# print("Similarity Matrix (Euclidean Distance):")
# print(similarity_matrix_df)

#Creating a knn-graph with neighbout 2 and euclidean algorithm
knn_graph = kneighbors_graph(selected_data, n_neighbors=k, mode='distance', metric=distance_algorithm)
G_knn = nx.from_scipy_sparse_array(knn_graph)

# Step 3: Assign Gaussian kernel weights to edges
def gaussian_kernel(distance, sigma=1.0):
    return math.exp(-distance**2 / (2 * sigma**2))

# Assign weights to edges in the graph
for u, v, d in G_knn.edges(data=True):
    distance = d['weight']  # The weight here is the Euclidean distance
    G_knn[u][v]['weight'] = gaussian_kernel(distance)

# #Display the graph with Gaussian Kernel Weights
# plt.figure(figsize=(8, 6))
# pos = nx.spring_layout(G_knn)  # positions for all nodes
# edge_weights = nx.get_edge_attributes(G_knn, 'weight')
# nx.draw(G_knn, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10, font_weight='bold')
# nx.draw_networkx_edge_labels(G_knn, pos, edge_labels=edge_weights)
# plt.title("K-Nearest Neighbors Graph with Gaussian Kernel Weights")
# plt.show()


#Graph partition using metis
num_clusters = 3
_, parts = metis.part_graph(G_knn, num_clusters)

# Assign partition results back to the nodes in the graph
for i, p in enumerate(parts):
    G_knn.nodes[i]['cluster'] = p

# #graph just before merging after partitioning
# for node in G_knn.nodes():
#     print(f"node:{node} -> {G_knn[node]} cluster:{G_knn.nodes[node].get('cluster')}")



#merging the clusters
def get_edge_weights(graph, node_pairs):
    return [graph[u][v]['weight'] for u, v in node_pairs]

#Find all the edges that connect two clusters
def connecting_edges(cluster_pair, graph):
    cluster_i, cluster_j = cluster_pair # Unpack the tuple to get the two clusters
    connecting_edges_list = [] # Initialize an empty list to store the connecting edges
    for u in cluster_i: # Iterate through each node in the first cluster
        for v in cluster_j: # For each node in the first cluster, iterate through each node in the second cluster
            if graph.has_edge(u, v):  # Check if there is an edge between node u and node v in the graph
                connecting_edges_list.append((u, v)) # If an edge exists, add it as a tuple (u, v) to the list of connecting edges
    return connecting_edges_list # Return the list of connecting edges


#Retrieve the subgraph corresponding to a cluster
def cluster_subgraph(graph, cluster):
    return graph.subgraph(cluster)

#Calculate the interconnectivity between two clusters
def calculate_interconnectivity(graph, cluster_i, cluster_j):
    edge_pairs = connecting_edges((cluster_i, cluster_j), graph)
    interconnectivity = np.sum(get_edge_weights(graph, edge_pairs))
    return interconnectivity

#Calculate the closeness between two clusters
def calculate_closeness(graph, cluster_i, cluster_j):
    cluster_i_subgraph = cluster_subgraph(graph, cluster_i)
    cluster_j_subgraph = cluster_subgraph(graph, cluster_j)
    internal_weights_i = get_edge_weights(cluster_i_subgraph, cluster_i_subgraph.edges())
    internal_weights_j = get_edge_weights(cluster_j_subgraph, cluster_j_subgraph.edges())
    closeness = np.sum(internal_weights_i) + np.sum(internal_weights_j)
    return closeness

#Merge two clusters and update the graph
def merge_clusters(graph, cluster_i, cluster_j):
    for node in cluster_j:
        graph.nodes[node]['cluster'] = graph.nodes[cluster_i[0]]['cluster']

#Find and merge the best clusters in the graph
def find_and_merge_best_clusters(graph, num_clusters):
    clusters = {} # Initialize an empty dictionary for clusters
    for node in graph.nodes(): # Iterate through each node in the graph
        cluster_id = graph.nodes[node]['cluster'] # Get the cluster id of the current node
        if cluster_id not in clusters: # Check if the cluster id is already a key in the clusters dictionary
            clusters[cluster_id] = [] # If not, initialize an empty list for this cluster id
        clusters[cluster_id].append(node) # Append the current node to the list of nodes for this cluster

    for node in graph.nodes():
        clusters[graph.nodes[node]['cluster']].append(node)
    
    best_score = 0
    clusters_to_merge = None
    
    # Consider all pairs of clusters for merging
    for cluster_i, cluster_j in itertools.combinations(clusters.keys(), 2):
        interconnectivity = calculate_interconnectivity(graph, clusters[cluster_i], clusters[cluster_j])
        closeness = calculate_closeness(graph, clusters[cluster_i], clusters[cluster_j])

        a = 1 # 'a' controls closeness and inerconnectivity. 1 means both are important. a < 1 means interconnectivity. a >1 closeness
        score = interconnectivity * (closeness ** a)
        
        if score > best_score:
            best_score = score
            clusters_to_merge = (cluster_i, cluster_j)

    if num_clusters is not None:
        if len(clusters) <= num_clusters:
            return False  # No more merging needed
    
    if clusters_to_merge: # Merge the clusters with the best score
        merge_clusters(graph, clusters[clusters_to_merge[0]], clusters[clusters_to_merge[1]])
        return True
    else:
        return False



#merging step
while find_and_merge_best_clusters(G_knn, num_clusters=2):
    pass  

node_colors = [G_knn.nodes[node]['cluster'] for node in G_knn.nodes()]

plt.figure(figsize=(10, 8),)
plt.suptitle(f"using {distance_algorithm} algorithm. k={k},parameters={selected_columns}")
nx.draw(G_knn, node_color=node_colors, with_labels=True)
plt.show()

