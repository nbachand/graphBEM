import numpy as np
import networkx as nx
import pandas as pd
from model.utils import *
from matplotlib import pyplot as plt

def getVFAlignedRectangles(X, Y, L):
    Xbar = X / L
    Ybar = Y / L

    return 2 / (np.pi * Xbar * Ybar) * (
        np.log(np.sqrt((1 + Xbar**2) * (1 + Ybar**2) / (1 + Xbar**2 + Ybar**2))) +
        Xbar * np.sqrt(1 + Ybar**2)  * np.arctan(Xbar / np.sqrt(1 + Ybar**2)) + 
        Ybar * np.sqrt(1 + Xbar**2)  * np.arctan(Ybar / np.sqrt(1 + Xbar**2)) -
        Xbar * np.arctan(Xbar) - 
        Ybar * np.arctan(Ybar)
        )

def getVFPerpRectanglesCommonEdge(X, Y, Z):
    H = Z / X
    W = Y / X
    return (1 / (np.pi * W)
    * (
        W * np.arctan(1 / W)
        + H * np.arctan(1 / H)
        - np.sqrt(H**2 + W**2) * np.arctan(1 / np.sqrt(H**2 + W**2))
        + 0.25 * np.log(
            (1 + W**2) * (1 + H**2)
            / (1 + W**2 + H**2)
            * (W**2 * (1 + W**2 + H**2) / ((1 + W**2) * (W**2 + H**2)))**W**2
            * (H**2 * (1 + H**2 + W**2) / ((1 + H**2) * (H**2 + W**2)))**H**2
        )
    )
)


class Radiation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["solveType"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        
        # Constants
        self.sigma = 5.67e-8
        self.storyHeight = 3

    def initialize(self, roomNode:nx.classes.coreviews.AtlasView):
        self.roomNode = dict(roomNode)
        surfaces = list(self.roomNode.keys())
        self.G = nx.Graph()
        # construct graphs based on radiation solve type
        if self.solveType == None:
            return
        self.G.add_nodes_from(surfaces)
        if self.solveType == "sky":
            self.G.add_node("sky")
            for surface in surfaces:
                self.G.add_edge(surface, "sky")
        if self.solveType == "room":
            for surface in surfaces:
                if surface != "RF":
                    self.G.add_edge(surface, "RF")
                if surface != "FL":
                    self.G.add_edge(surface, "FL")
        nx.draw(self.G, with_labels=True)

        # consturct radiation graph for the room (roof to floor, including via walls)
        for n, d in self.G.nodes(data=True):
            if n == "sky":
                epislon = 1 # This is not the true emmisivity (using W to specify sky intensity) but ignores reflected radiation
                A = 1 # doesn't matter sice epsilon = 1
            else:
                d["T_index"] = self.roomNode[n]["nodes"].getSideIndex(n, reverse = True) # reversed because front is in other room
                wall = self.roomNode[n]["wall"]
                epislon = wall.alpha # opaque, diffuse, gray surface
                d["X"] = wall.X  
                d["Y"] = wall.Y
                A = d["X"] * d["Y"]
            d["boundaryResistance"] = (1 - epislon) / (epislon * A)

        for i, j, d in self.G.edges(data=True):
            #don't use properties of "sky" node
            if i == "sky":
                i = j
            elif j == "sky":
                j = i
            #calc radiance resistance
            X = self.G.nodes[i]["X"]
            Y = self.G.nodes[i]["Y"]
            A = X * Y
            if self.solveType == "sky":
                F = 1
            elif set([i, j]) == set(["RF", "FL"]):
                if X != self.G.nodes[j]["X"] or Y != self.G.nodes[j]["Y"]:
                    raise Exception("Dimmensions of roof and floor do not match")
                F = getVFAlignedRectangles(X, Y, self.storyHeight)
            else:
                Z = [self.G.nodes[j]["X"], self.G.nodes[j]["Y"]]
                if X not in Z:
                    raise Exception("Dimmension along seam of {i} and {j} do not match")
                Z.remove(X)
                F = getVFPerpRectanglesCommonEdge(X, Y, Z[0])  
            d["radianceResistance"] = (A * F) ** -1
        plt.figure()
        self.A = graphToSysEqnKCL(self.G)

    def timeStep(self, solarGain = 0):
        if self.solveType == None:
            return pd.Series()
        bR = pd.Series(0.0, index = self.A.index)
        Eb = pd.Series(0.0, index = self.A.index)
        A = pd.Series(0.0, index = self.A.index)
        for n, d in self.G.nodes(data=True):
            if n == "sky":
                Eb[n] = solarGain
                A[n] = 1
            else: 
                wall = self.roomNode[n]["wall"]
                T = wall.T_prof[d["T_index"]]
                Eb[n] = self.sigma * T**4
                A[n] = d['X'] * d['Y']
            bR[n] = d["boundaryResistance"]
        J = np.linalg.solve(self.A, Eb)
        J = pd.Series(J, index = self.A.index)
        q = (J - Eb) / bR
        E = q / A #are averaged radiative heat flux
        return E