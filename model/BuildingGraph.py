import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from copy import deepcopy

class BuildingGraph:
    def __init__(self, connectivityMatrix:np.array =  np.array([[]]), roomList:list = []):
        self.connectivityMatrix = connectivityMatrix
        self.roomList = roomList
        self.n = connectivityMatrix.shape[0]
        self.m = connectivityMatrix.shape[1]
        self.G = nx.Graph()
        self.G.add_nodes_from(roomList)
        for i in range(self.n): # solved nodes
            for j in range(i, self.m): # forcing nodes
                if connectivityMatrix[i, j] != 0:
                    self.G.add_edge(
                        roomList[i][0], 
                        roomList[j][0], 
                        weight = connectivityMatrix[i, j],
                        front = roomList[i][0],
                        back = roomList[j][0]
                        )

    def draw(self):
        plt.figure()
        # nx.draw(self.G, with_labels=True)

        elarge = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] > 0.5]
        esmall = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] <= 0.5]

        pos = nx.spring_layout(self.G, seed=7)  # positions for all nodes - seed for reproducibility

        # nodes
        nx.draw_networkx_nodes(self.G, pos, node_size=700)

        # edges
        nx.draw_networkx_edges(self.G, pos, edgelist=elarge, width=6)
        nx.draw_networkx_edges(
            self.G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
        )

        # node labels
        nx.draw_networkx_labels(self.G, pos, font_size=20, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()

    def updateEdges(self, properties: dict, nodes = None, edges = None):
        """
        nodes updates all edges with the node in the list while edges specifies the exact edges to update
        """
        for i, j, d in self.G.edges(data=True):
            if nodes is None and edges is None:
                d.update(deepcopy(properties))
            if nodes is not None and (i in nodes or j in nodes):
                d.update(deepcopy(properties))
            if edges is not None and ((i, j) in edges or (j, i) in edges):
                d.update(deepcopy(properties))

    def updateNodes(self, properties: dict, nodes = None):
        for n, d in self.G.nodes(data=True):
            if nodes is None or n in nodes:
                d.update(deepcopy(properties))