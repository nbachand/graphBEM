import numpy as np
import pandas as pd
import networkx as nx
from model.utils import *
from model.BuildingGraph import draw

EXTERIOR_EMISSIVITY = 0.9

def incident_longwave(horizontal_ir, air_temperature, tilt, sky_model="isotropic"):
    """Unobstructed sky irradiance; surroundings are at outdoor air temperature.

    EnergyPlus sky/air split: Engineering Reference 22.2, Outside Surface Heat
    Balance, External Longwave Radiation. Input IR already includes sky emissivity.
    """
    if sky_model not in {"isotropic", "energyplus"}:
        raise ValueError("Unknown sky model")
    if not 0 <= tilt <= 180:
        raise ValueError("Face tilt must be in [0, 180]")
    view = (1 + np.cos(np.radians(tilt))) / 2
    if sky_model == "energyplus":
        view *= np.sqrt(view)
    return view*horizontal_ir + (1-view)*5.67e-8*air_temperature**4


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
        if not expected_kwards <= set(kwargs) or set(kwargs) - expected_kwards - {"storyHeight", "emissivity", "sky_model"}:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        
        self.sky_model = kwargs.get("sky_model", "precombined")
        if self.sky_model not in {"precombined", "isotropic", "energyplus"}:
            raise ValueError("Unknown sky model")
        self.horizontalIR = None
        self.airTemperature = None
        # Constants
        self.sigma = 5.67e-8
        self.storyHeight = kwargs.get("storyHeight", 3)

    def initialize(self, roomNode:nx.classes.coreviews.AtlasView, solarGain=0, drawGraphs=False, longwaveGain=0):
        self.solarGain = solarGain  # incident shortwave, W/m2
        self.longwaveGain = longwaveGain  # incident sky + ground longwave, W/m2
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
                alpha = 1 # This is not the true absorptivity (using W to specify sky intensity) but ignores reflected radiation (e.g., the sky doesnt reflect)
                d["A"] = 1 # doesn't matter since epsilon = 1
                # d["epsilon_over_alpha"] = 1 # dont think this is used
            else:
                wall = self.roomNode[n]["wall"]
                alpha = (getattr(self, "emissivity", wall.absorptivity)
                         if self.solveType == "room" else wall.absorptivity)
                if not np.isfinite(alpha) or not 0 < alpha <= 1:
                    raise ValueError("Surface absorptivity must be in (0, 1]")
                d["X"] = wall.X # dimension used in view factor  
                d["Y"] = wall.Y # dimension used in view factor
                d["A"] = d["X"] * d["Y"] * self.roomNode[n]["weight"] # true area
                d["T_index"] = self.roomNode[n]["nodes"].getSideIndex(n, reverse = True) # reversed because front is in other room
                if d["T_index"] == 999:
                    d["T_index"] = 0 # arbitrary for partition walls which should be symetrical
                    d["A"] *= 2
                if self.solveType == "sky":
                    d["epsilon_over_alpha"] = EXTERIOR_EMISSIVITY / alpha # emmisivity to sky is ~0.9
                else:
                    d["epsilon_over_alpha"] = 1
            d["boundaryResistance"] = (1 - alpha) / (alpha * d["A"])

        for i, j, d in self.G.edges(data=True):
            if self.solveType == "sky":
                surface = j if i == "sky" else i
                exchange_area = self.G.nodes[surface]["A"]
            elif {i, j} == {"RF", "FL"}:
                roof, floor = self.G.nodes["RF"], self.G.nodes["FL"]
                if roof["A"] != floor["A"]:
                    raise ValueError("Areas of roof and floor do not match")
                F = getVFAlignedRectangles(roof["X"], roof["Y"], self.storyHeight)
                exchange_area = roof["A"] * F
            else:
                # Always evaluate wall -> horizontal surface. The wall's area
                # includes its multiplicity (and both faces for self-loops).
                # The same A*F is used in both directions, enforcing reciprocity.
                horizontal = i if i in {"RF", "FL"} else j
                wall = j if horizontal == i else i
                w, h = self.G.nodes[wall], self.G.nodes[horizontal]
                dimensions = [h["X"], h["Y"]]
                matches = [k for k, length in enumerate(dimensions) if np.isclose(w["X"], length)]
                if not matches:
                    raise ValueError(f"No shared edge between {wall} and {horizontal}")
                dimensions.pop(matches[0])
                F = getVFPerpRectanglesCommonEdge(w["X"], w["Y"], dimensions[0])
                exchange_area = w["A"] * F
            d["radianceResistance"] = 1 / exchange_area
        if drawGraphs:
            draw(self.G, weight = "radianceResistance")
        self.A = graphToSysEqnKCL(self.G)

    def timeStep(self):
        if self.solveType == None:
            return pd.Series()
        if self.solveType == "sky":
            # Spectral bands must remain separate: sky emissivity is already
            # represented by the weather-file infrared irradiance.
            result = {}
            for n, d in self.G.nodes(data=True):
                if n == "sky":
                    continue
                wall = self.roomNode[n]["wall"]
                incoming = self.longwaveGain
                if self.sky_model != "precombined":
                    if self.horizontalIR is None or self.airTemperature is None:
                        raise ValueError("Sky model requires horizontal IR and outdoor air temperature")
                    side = "front" if d["T_index"] == 0 else "back"
                    incoming = incident_longwave(self.horizontalIR, self.airTemperature,
                                                getattr(wall.tilt, side), self.sky_model)
                result[n] = (wall.absorptivity*self.solarGain + EXTERIOR_EMISSIVITY*
                             (incoming-self.sigma*wall.T_prof[d["T_index"]]**4))
            return pd.Series(result)
        Eb = np.array([self.sigma * self.roomNode[n]["wall"].T_prof[d["T_index"]]**4
                       for n, d in self.G.nodes(data=True)])
        J = pd.Series(np.linalg.solve(self.A, Eb), index=self.A.index)
        # Edge heat flows remain finite when emissivity=1 and surface R=0.
        # Equal and opposite edge powers also conserve enclosure radiation.
        power = pd.Series(0.0, index=J.index)
        for i, j, edge in self.G.edges(data=True):
            q = (J[j] - J[i]) / edge["radianceResistance"]
            power[i] += q
            power[j] -= q
        return power / pd.Series({n: d["A"] for n, d in self.G.nodes(data=True)})
