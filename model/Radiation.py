import numpy as np
import pandas as pd
import networkx as nx
from model.utils import *
from model.BuildingGraph import draw

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

    def initialize(self, roomNode:nx.classes.coreviews.AtlasView, drawGraphs = False):
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
            for n in self.G.nodes:
                if n != "RF":
                    self.G.add_edge(n, "RF")
                if n != "FL":
                    self.G.add_edge(n, "FL")

        # assign properties to the radiation graph
        for n, d in self.G.nodes(data=True):
            if n == "sky":
                alpha = 1 # This is not the true absorptivity (using W to specify sky intensity) but ignores reflected radiation
                d["A"] = 1 # doesn't matter sice epsilon = 1
                d["alpha_over_epsilon"] = 1
            else:
                wall = self.roomNode[n]["wall"]
                alpha = wall.absorptivity # opaque, diffuse, gray surface
                d["X"] = wall.X # dimension used in view factor  
                d["Y"] = wall.Y # dimension used in view factor
                d["A"] = d["X"] * d["Y"] * self.roomNode[n]["weight"] # true area
                d["T_index"] = self.roomNode[n]["nodes"].getSideIndex(n, reverse = True) # reversed because front is in other room
                if d["T_index"] == 999:
                    d["T_index"] = 0 # arbitrary for partition walls which should be symetrical
                    d["A"] *= 2
                if self.solveType == "sky":
                    d["epsilon_over_alpha"] = 0.9 / alpha #radiation from roof is emmitted to sky with emissivity of 0.9 (different than to other surfaces do to nature of atmospheric abosrption)
                else:
                    d["epsilon_over_alpha"] = 1
            d["boundaryResistance"] = (1 - alpha) / (alpha * d["A"])

        for i, j, d in self.G.edges(data=True):
            #don't use properties of "sky" node
            if i == "sky":
                i = j
            elif j == "sky":
                j = i
            #calc radiance resistance
            X = self.G.nodes[i]["X"]
            Y = self.G.nodes[i]["Y"]
            if self.solveType == "sky":
                F = 1
            elif set([i, j]) == set(["RF", "FL"]):
                if self.G.nodes[i]["A"] != self.G.nodes[j]["A"]:
                    raise Exception("Areas of roof and floor do not match")
                F = getVFAlignedRectangles(X, Y, self.storyHeight)
            else:
                Z = [self.G.nodes[j]["X"], self.G.nodes[j]["Y"]]
                if X not in Z:
                    raise Exception("Dimmension along seam of {i} and {j} do not match")
                Z.remove(X)
                F = getVFPerpRectanglesCommonEdge(X, Y, Z[0]) 
            d["radianceResistance"] = (self.G.nodes[i]["A"] * F) ** -1
        if drawGraphs:
            draw(self.G, weight = "radianceResistance")
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
                Eb[n] *= d["epsilon_over_alpha"] # use a modified black body radiation if epsilon != alpha
                A[n] = d['A']
            bR[n] = d["boundaryResistance"]
        J = np.linalg.solve(self.A, Eb)
        J = pd.Series(J, index = self.A.index)
        q = (J - Eb) / bR
        E = q / A #area averaged radiative heat flux
        return E