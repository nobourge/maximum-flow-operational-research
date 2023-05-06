#
import networkx as nx
from matplotlib import pyplot as plt


class GraphVisualizer:
    def __init__(self, graph):
        # graph is a list of lists
        # graph[i] is a list of neighbors of node i
        # graph[i][j] is the j-th neighbor of node i

        self.graph = graph

    def draw(self):

        # Create a graph
        G = nx.Graph()

        # Add nodes
        for node in range(len(self.graph)):
            G.add_node(node)

        # Add edges
        for node in range(len(self.graph)):
            for neighbor in self.graph[node]:
                G.add_edge(node, neighbor)

        # Draw the graph
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()
