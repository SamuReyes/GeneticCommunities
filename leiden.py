from nsga import NSGA
from moga import MOGA
import networkx as nx
import matplotlib.pyplot as plt
import random

def Modularity(graph):
    """
    Calculates the modularity of the graph
    """

    # Step 1: Extract the node attributes into a dictionary
    node_community_map = nx.get_node_attributes(graph, 'community')

    # Step 2: Invert the dictionary
    communities = {}
    for node, community in node_community_map.items():
        if community not in communities:
            communities[community] = []
        communities[community].append(node)

    # Convert the values to a list of sets (or lists)
    community_list = list(map(set, communities.values()))

    # Step 3: Calculate modularity
    modularity = nx.algorithms.community.modularity(graph, community_list)
    
    return modularity



def is_connected_subgraph(graph, nodes):
    """
    Check if the subgraph formed by 'nodes' in 'graph' is connected.
    """
    subgraph = graph.subgraph(nodes)
    return nx.is_connected(subgraph)



def find_best_community_for_node(graph, max_modularity, node):
    """
    Find the best community for the given node.
    """
    neighbors = list(graph.neighbors(node))

    # If the node has no neighbors, we can't move it anywhere
    if len(neighbors) == 0:
        return graph, max_modularity

    neighbor_communities = set()
    for neighbor in neighbors:

        neighbor_community = graph._node[neighbor]["community"]
        
        # try:
        #     neighbor_community = graph._node[neighbor]["community"]
        # except:
        #     with open ("log.txt", "w") as file:
        #         file.write(str(vars(graph)))

        neighbor_communities.add(neighbor_community)

    # If the node is already in a community by itself, we can't move it anywhere
    if len(neighbor_communities) == 0:
        return graph, max_modularity

    # If the node is already in a community with all of its neighbors, we don't need to move it
    if len(neighbor_communities) == 1 and graph._node[node]["community"] in neighbor_communities:
        return graph, max_modularity

    neighbor_communities = list(neighbor_communities)          
    neighbor_communities = sorted(neighbor_communities)
    
    # Try moving the node to each of its neighbor's communities
    saved_graph = graph.copy()
    for community in neighbor_communities:
        graph._node[node]["community"] = community

        # Check if the resulting community is a connected subgraph
        community_nodes = [n for n, attr in graph.nodes(data=True) if attr["community"] == community]
        if is_connected_subgraph(graph, community_nodes):
            modularity = Modularity(graph)
            if modularity > max_modularity:
                max_modularity = modularity
                saved_graph = graph.copy()

    return saved_graph, max_modularity



def aggregate_network(graph):
    """
    Aggregates the network by creating a new graph where each node represents a community.
    """
    
    # Step 1: Create a mapping from each node to its community
    community_map = {node: data['community'] for node, data in graph.nodes(data=True)}

    # Step 2: Create a new graph where each node represents a community
    aggregated_graph = nx.Graph()

    # Step 3: Add nodes to the aggregated graph, one for each community
    for community in set(community_map.values()):
        aggregated_graph.add_node(community)

    # Step 4: Add edges between communities in the aggregated graph
    for u, v in graph.edges():
        community_u = community_map[u]
        community_v = community_map[v]
        if community_u != community_v:
            aggregated_graph.add_edge(community_u, community_v)

    return aggregated_graph



def Leiden(graph, refinement_probability=0.2):
    """
    Implementation of the Leiden algorithm for community detection.
    """

    # Intialize each node to its own community
    for node in graph.nodes():
        graph._node[node]["community"] = node
    max_modularity = -1

    # Iteratively improve modularity
    while True:

        # Step 1: Local moving of nodes
        improved = False
        for node in graph.nodes():
            graph, modularity = find_best_community_for_node(graph, max_modularity, node)
            if modularity > max_modularity:
                improved = True
                max_modularity = modularity

        # Step 2: Refinement phase
        refined = False
        for node in graph.nodes():
            random_number = random.uniform(0, 1)
            if random_number <= refinement_probability:
                graph, modularity = find_best_community_for_node(graph, max_modularity, node)
                
                if modularity > max_modularity:
                    refined = True
                    max_modularity = modularity

        # Step 3: Network aggregation

        graph = aggregate_network(graph)
        
        if not improved and not refined:
            break
                    
    return graph, modularity