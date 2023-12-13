import numpy as np
import scipy.linalg as sp_linalg
from matplotlib import pyplot as plt
from scipy.integrate import trapz
import networkx as nx
from copy import deepcopy
import pandas as pd

def getEquivalentTimeSeries(x, tSeries):
    if isinstance(x, float):
        x = np.ones_like(tSeries) * x
    elif x is None:
        x =  np.zeros_like(tSeries)
    elif len(x) != len(tSeries):
        raise Exception("x and tSeries must be the same length")
    return x

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

# Function to build the system of equations
def graphToSysEqnKCL(graph):
    A = pd.DataFrame(0.0, graph.nodes, columns=graph.nodes)

    # Build equations based on KCL for each node (except the reference node)
    for n, d in graph.nodes(data=True):
        # Sum of currents entering the node equals the sum of currents leaving the node (KCL)
        A[n][n] += 1
        for e in graph[n]:
            A[e][n] = -d["boundaryResistance"] / graph[n][e]['weight']
            A[n][n] += d["boundaryResistance"] / graph[n][e]['weight']

    return A



class BuildingSimulation():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["delt", "simLength", "Tout", "radG", "Tfloor"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        self.t = 0 #time (seconds)
        self.hour = 0 #time (hours)
        self.times = np.arange(0, self.simLength + self.delt, self.delt)
        self.hours = self.times / 60 / 60
        self.N = len(self.times)
        self.Tout = getEquivalentTimeSeries(self.Tout, self.times)
        self.radGF = getEquivalentTimeSeries(self.radG, self.times)

    def initialize(self, bG):
        self.bG = bG
        for n, d in self.bG.G.nodes(data=True):
            r = RoomSimulation(**d["room_kwargs"])
            v = VentilationSimulation(**d["vent_kwargs"])
            r.initialize(self.delt)
            if n == "OD":
                r.Tint = self.Tout[0]
            elif n == "FL":
                r.Tint = self.Tfloor

            Tints = np.zeros(self.N) # initializing interior air temp vector
            Tints[0] = r.Tint
    
            d.update({"room": r, 
                      "vent": v,
                      "Tints": Tints,
                      "Vnvs": np.zeros(self.N), # initializing ventilation energy vector
                      "Ef": 0,
                      })
        for i, j, d in self.bG.G.edges(data=True):
            w = WallSimulation(**d["wall_kwargs"])
            Tff = self.bG.G.nodes[d["front"]]["room"].Tint
            Tfb = self.bG.G.nodes[d["back"]]["room"].Tint
            w.initialize(self.delt, Tff, Tfb)

            T_profs = np.zeros((w.n + 2, self.N)) # intializing matrix to store temperature profiles
            T_profs[:, 0] = w.getWallProfile(Tff, Tfb)

            d.update({
                "wall": w,
                "T_profs": T_profs,
                })

    def run(self):
        for c in range(1, self.N):
            self.t = self.times[c]
            self.hour = self.t / 60 / 60

            # Simulation logic
            # Solve Walls
            for i, j, d in self.bG.G.edges(data=True):
                Ef = d["wall"].timeStep(self.bG.G.nodes[d["front"]]["room"].Tint, self.bG.G.nodes[d["back"]]["room"].Tint)
                self.bG.G.nodes[d["front"]]["Ef"] += Ef.front * d["weight"]
                self.bG.G.nodes[d["back"]]["Ef"] += Ef.back * d["weight"]
                d["T_profs"][:,c] = d["wall"].T_prof

            # Solve Rooms
            for n, d in self.bG.G.nodes(data=True):
                if n == "OD":
                    d["room"].Tint = self.Tout[c] # if outdoors, just use outdoor temp
                elif n == "FL":
                    d["room"].Tint = self.Tfloor # if floor, just use floor temp
                else:
                    Evt = d["vent"].timeStep(self.t, Tint = d["room"].Tint, Tout = self.Tout[c])
                    d["Vnvs"][c] = d["vent"].Vnv
                    d["room"].timeStep(d["Ef"], Evt)
                d["Tints"][c] = d["room"].Tint
                d["Ef"] = 0 #resetting Ef for next time step


    def runOld(self):
        self.wall = self.bG.G.edges['R', 'R']["wall"]
        self.room = self.bG.G.nodes['R']["room"]
        self.vent = self.bG.G.nodes['R']["vent"]
        Tints = np.zeros(self.N) # initializing interior air temp vector
        Vnvs = np.zeros(self.N) # initializing ventilation energy vector
        Tint = self.room.Tint
        Tints[0] = Tint

        T_profs = np.zeros((self.wall.n + 2, self.N)) # intializing matrix to store temperature profiles
        T_profs[:, 0] = self.wall.getWallProfile(Tint, Tint)
        nWalls = 3 #number of walls

        for i in range(1, self.N):
            self.t = self.times[i]
            self.hour = self.t / 60 / 60

            # Simulation logic
            Ef = self.wall.timeStep(self.room.Tint, self.room.Tint)
            Evt = self.vent.timeStep(self.t, Tint = self.room.Tint, Tout = self.Tout[i])
            self.room.timeStep(nWalls * (Ef.front + Ef.back), Evt)

            T_profs[:,i] = self.wall.T_prof
            Tints[i] = self.room.Tint
            Vnvs[i] = self.vent.Vnv
        return Tints, T_profs, Vnvs


class WallFlux:
    def __init__(self):
        self.front = None
        self.back = None

class RoomSimulation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["T0", "V", "Eint"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        # Constants
        self.rho = 1.225 #air density
        self.Cp = 1005  #specific heat capacity for air

    def initialize(self, delt):
        self.Tint = self.T0

        # Scaling factors
        self.lambda_int = delt / (self.rho * self.Cp * self.V)

    def timeStep(self, Ef, Evt):
        self.Tint = self.Tint + self.lambda_int * (Ef + self.Eint + Evt)
        return self.Tint
        


class WallSimulation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set([])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        # Constants
        self.rhof = 2300 #density of fabric
        self.Cf = 750 #specific heat capacity of fabric
        self.kf = 0.8 #thermal conductivity of fabric
        self.Af = 90 #fabric area
        self.th = 0.10 #fabric thickness 
        self.h = 4 #fabric convection coefficient
        self.delx = 0.010 #spatial discretization size
        self.x = np.arange(0, self.th + self.delx, self.delx)
        self.alpha = 0.7 #fabric absorptivity

    def initialize(self, delt, TfF, TfB):
        # Scaling factors
        self.lambda_val = (self.kf * delt) / (self.rhof * self.Cf * self.delx**2)
        self.lambda_bound = self.kf / (self.h * self.delx)

        # Wall setup
        self.n = round(self.th / self.delx) - 1
        r = np.zeros(self.n)
        r[0] = 1 - 2 * self.lambda_val
        r[1] = self.lambda_val
        r[-1] = self.lambda_val

        A_matrix = sp_linalg.toeplitz(r)
        A_matrix[0, 0] += self.lambda_val * self.lambda_bound / (1 + self.lambda_bound)
        A_matrix[-1, -1] = A_matrix[0, 0]
        A_matrix[-1, 0] = 0
        A_matrix[0, -1] = 0

        self.A = A_matrix

        self.b = np.zeros(self.n)
        self.T_prof = np.linspace(TfF, TfB, self.n + 2) #create a uniform temperature profile between Tff and Tfb of length n
        self.T = self.T_prof[1:-1] #remove the boundary temperatures from the temperature profile

        self.GF = 0 #front radiative gain
        self.GB = 0 #back radiative gain

    def timeStep(self, TintF, TintB):
        self.b[0] = self.lambda_val * (TintF + self.GF * self.alpha/self.h) / (1 + self.lambda_bound)
        self.b[-1] = self.lambda_val * (TintB + self.GB * self.alpha/self.h) / (1 + self.lambda_bound)
        self.T = np.dot(self.A, self.T) + self.b
        self.T_prof = self.getWallProfile(TintF, TintB)

        Ef = WallFlux()
        Ef.front = self.Af * (self.T_prof[1] - self.T_prof[0]) / self.delx
        Ef.back = self.Af * (self.T_prof[-2] - self.T_prof[-1]) / self.delx
        return Ef

    def getWallProfile(self, TintF, TintB):
        T_prof = np.zeros(self.n + 2)
        T_prof[1:-1] = self.T
        T_prof[0] = self.get_Tf(self.T[0], TintF, self.lambda_bound)
        T_prof[-1] = self.get_Tf(self.T[-1], TintB, self.lambda_bound)
        return T_prof

    def get_Tf(self, T1, Tint, lambda_bound):
        return (lambda_bound * T1 + Tint) / (1 + lambda_bound)

class VentilationSimulation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(['H', 'W', "ventType", "alphas", "As", "Ls"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        
        # Constants
        self.rho = 1.225 #air density
        self.Cp = 1005  #specific heat capacity for air
        self.Vnv = None

        self.initialize()

    def initialize(self):
        if self.alphas is None:
            self.Cds = None
            return
        self.Cds = np.zeros(len(self.alphas))
        for i, alpha in enumerate(self.alphas):
            self.Cds[i] = self.get_Cd(np.radians(alpha))
        
    def get_Wpivot(self, z, alpha):
        h = self.H * (1 - np.cos(alpha))
        Wpivot = np.ones_like(z) * self.W
        zpivot = z[z > h]
        Wpivot[z > h] = (1 / self.W**2 + 1 / (2 * (self.H - zpivot) * np.tan(alpha) + np.sin(alpha) * self.W**2))**(-1/2)
        return Wpivot

    def get_Aeff(self, alpha):
        z = np.linspace(0, self.H, 1000)
        Wpivot = self.get_Wpivot(z, alpha)
        return trapz(Wpivot, z)

    def get_Cd(self, alpha):
        Cd0 = 0.611
        return self.get_Aeff(alpha) / (self.H * self.W) * Cd0

    def get_Vnvi(self, Cdi, Ai, Li, Tint, Tout):
        g = 9.8
        return Cdi * Ai * (2 * g * Li * np.abs((Tint - Tout) / Tout))**0.5

    def get_Vnv(self, Tint, Tout, t):
        Vnv = np.zeros((self.Cds.size, t.size))
        hours = t / 60 / 60
        day_hours = np.remainder(hours, 24)
        if isinstance(Tint, float):
            Tint = np.ones_like(hours) * Tint
        if isinstance(Tout, float):
            Tout = np.ones_like(hours) * Tout
        Tint = Tint[(day_hours < 7) | (day_hours > 12 + 7)]
        Tout = Tout[(day_hours < 7) | (day_hours > 12 + 7)]
        for i in range(len(self.As)):
            Vnv_7to7 = self.get_Vnvi(self.Cds[i], self.As[i], self.Ls[i], Tint, Tout)
            Vnv_i = Vnv[[i], :]
            Vnv_i[(day_hours < 7) | (day_hours > 12 + 7)] = Vnv_7to7
            Vnv[i, :] = Vnv_i
        return Vnv
        
    def qToEvt(self, q, Tout, Tint):
        return self.rho * self.Cp * q * (Tout - Tint)

    def timeStepHWP1(self, t):
        hour = t / 60 / 60
        Evt = -500
        if (hour % 24 > 7) and (hour % 24 < 19):  # daytime
            return Evt
        return 0
    
    def timeStepHWP4(self, t, Tint = 0, Tout = 0):
        self.Vnv = np.sum(self.get_Vnv(Tint, Tout, t), axis=0) #summing across windows
        Evt = self.qToEvt(self.Vnv, Tout, Tint)
        return Evt
    
    def timeStep(self, *args, **kwargs):
        if self.ventType == "HWP1":
            return self.timeStepHWP1(*args)
        if self.ventType == "HWP4":
            return self.timeStepHWP4(*args, **kwargs)
        if self.ventType == "None":
            return 0
    

class BuildingGraph:
    def __init__(self, connectivityMatrix:np.array, roomList:list):
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

    def updateAllEdges(self, properties: dict):
        for i, j, d in self.G.edges(data=True):
            d.update(deepcopy(properties))

    def updateAllNodes(self, properties: dict):
        for n, d in self.G.nodes(data=True):
            d.update(deepcopy(properties))

class Radiation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["solveRooms"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        
        # Constants
        self.sigma = 5.67e-8

    def initialize(self, roomName, buildingGraph:nx.classes.graph.Graph, delt):
        self.delt = delt
        self.name = roomName
        self.roomNode = buildingGraph[self.name]
        X = 4
        Y = 4
        A = X * Y
        if self.name == "RF":
            wallList = [
                ("sun", {"area": A}), 
                ("SS", {"area": A}),
                # ("DR", {"area": A}),
                # ("CV", {"area": A}),
                # ("CR", {"area": A}),
                ]
            F = 1
            W = (A * F) ** -1
            # connectivityMatrix = np.array([
            # [0, W, W, W, W],
            # [W, 0, 0, 0, 0],
            # [W, 0, 0, 0, 0],
            # [W, 0, 0, 0, 0],
            # [W, 0, 0, 0, 0],
            # ])
            connectivityMatrix = np.array([
            [0, W],
            [W, 0],
            ])
        elif self.name == "DR":
            wallList = [
                ("OD", {"area": A}), 
                ("CR", {"area": A}),
                ("CV", {"area": A})
                ]
            F = 1/2
            W = (A * F) ** -1
            connectivityMatrix = np.array([
            [0, W, W],
            [W, 0, W],
            [W, W, 0],
            ])
        elif self.name in self.solveRooms:
            wallList = [
                ("OD", {"area": A}),
                ("FL", {"area": A})
                ]
            L = 3
            F = getVFAlignedRectangles(X, Y, L)
            W = (A * F) ** -1
            connectivityMatrix = np.array([
            [0, W],
            [W, 0],
            ])
        self.bG = BuildingGraph(connectivityMatrix, wallList)
        self.bG.updateAllEdges({"radiance": 0})

        # assign properties to raidation graph
        for n, d in self.bG.G.nodes(data=True):
            if n == "sun":
                epislon = 1
            else:
                if n == self.roomNode[n]["front"]:
                    d["T_index"] = -1 # reversed because front is in other room
                elif n == self.roomNode[n]["back"]:
                    d["T_index"] = 0 #reversed because back is in other room
                else:
                    raise Exception("Wall front/back has been missassigned")
                
                wall = self.roomNode[n]["wall"]
                epislon = wall.alpha # opaque, diffuse, gray surface
            d["boundaryResistance"] = (1 - epislon) / (epislon * d["area"])
        self.A = graphToSysEqnKCL(self.bG.G)

    def timeStep(self, solarGain = 0):
        bR = pd.Series(0.0, index = self.A.index)
        Eb = pd.Series(0.0, index = self.A.index)
        for n, d in self.bG.G.nodes(data=True):
            if n == "sun":
                Eb[n] = solarGain
            else: 
                wall = self.roomNode[n]["wall"]
                T = wall.T_prof[d["T_index"]]
                Eb[n] = self.sigma * T**4
                print(T)
            bR[n] = d["boundaryResistance"]
        J = np.linalg.solve(self.A, Eb)
        J = pd.Series(J, index = self.A.index)
        q = (J - Eb) / bR
        return  q * self.delt# radiative gain for each node