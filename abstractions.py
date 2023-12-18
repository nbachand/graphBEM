import numpy as np
import scipy.linalg as sp_linalg
from matplotlib import pyplot as plt
from scipy.integrate import trapz
import networkx as nx
from copy import deepcopy
import pandas as pd

def getEquivalentTimeSeries(x, tSeries):
    if isinstance(x, float) or isinstance(x, int):
        x = np.ones_like(tSeries) * x
    elif x is None:
        x =  np.zeros_like(tSeries)
    elif len(x) != len(tSeries):
        raise Exception("x and tSeries must be the same length")
    return x

def getVFAlignedRectangles(X, Y, L):
    if L == -1:
        return 1
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
            A[e][n] = -d["boundaryResistance"] / graph[n][e]['radianceResistance']
            A[n][n] += d["boundaryResistance"] / graph[n][e]['radianceResistance']

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
        self.radDamping = self.delt / (1 + self.delt)# damping factor for radiation

    def initialize(self, bG):
        self.bG = bG
        for n, d in self.bG.G.nodes(data=True):
            r = RoomSimulation(**d["room_kwargs"])
            v = VentilationSimulation(**d["vent_kwargs"])
            r.initialize(self.delt)
            if n == "OD" or n == "RF":
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
            if d["front"] == "FL":
                w.h.front = 1e6
            elif d["back"] == "FL":
                w.h.back = 1e6
            w.initialize(self.delt, Tff, Tfb)

            T_profs = np.zeros((w.n + 2, self.N)) # intializing matrix to store temperature profiles
            T_profs[:, 0] = w.getWallProfile(Tff, Tfb)

            radEApplied = WallFlux()
            radECalc = WallFlux()
            for radE in [radEApplied, radECalc]:
                radE.front = np.zeros(self.N)
                radE.back = np.zeros(self.N)

            d.update({
                "wall": w,
                "T_profs": T_profs,
                "radEApplied": radEApplied,
                "radECalc": radECalc,
                })
        for n, d in self.bG.G.nodes(data=True):
            rad = Radiation(**d["rad_kwargs"])
            rad.initialize(self.bG.G[n])
            d.update({"rad": rad})


    def run(self):
        for c in range(1, self.N):
            self.t = self.times[c]
            self.hour = self.t / 60 / 60

            # Simulation logic
            # Solve Radiation
            for n, d in self.bG.G.nodes(data=True):
                q = d["rad"].timeStep(self.radGF[c])
                q = q.dropna()
                for wall, qWall in q.items():
                    if wall == "sun":
                        continue
                    if n == self.bG.G.edges[n, wall]["front"]:
                        qOld = self.bG.G.edges[wall, n]["wall"].qradF
                        self.bG.G.edges[(n, wall)]["radECalc"].front[c] = qWall
                        qWall = (1 - self.radDamping) * qWall + self.radDamping * qOld
                        self.bG.G.edges[(n, wall)]["radEApplied"].front[c] = qWall
                        self.bG.G.edges[wall, n]["wall"].qradF = qWall
                    elif n == self.bG.G.edges[n, wall]["back"]:
                        qOld = self.bG.G.edges[wall, n]["wall"].qradB
                        self.bG.G.edges[(n, wall)]["radECalc"].back[c] = qWall
                        qWall = (1 - self.radDamping) * qWall + self.radDamping * qOld
                        self.bG.G.edges[(n, wall)]["radEApplied"].back[c] = qWall
                        self.bG.G.edges[wall, n]["wall"].qradB = qWall
                    else:
                        raise Exception("Wall front/back has been missassigned")


            # Solve Walls
            for i, j, d in self.bG.G.edges(data=True):
                Ef = d["wall"].timeStep(self.bG.G.nodes[d["front"]]["room"].Tint, self.bG.G.nodes[d["back"]]["room"].Tint)
                self.bG.G.nodes[d["front"]]["Ef"] += Ef.front * d["weight"]
                self.bG.G.nodes[d["back"]]["Ef"] += Ef.back * d["weight"]
                d["T_profs"][:,c] = d["wall"].T_prof

            # Solve Rooms
            for n, d in self.bG.G.nodes(data=True):
                if n == "OD" or n == "RF":
                    d["room"].Tint = self.Tout[c] # if outdoors, just use outdoor temp
                elif n == "FL":
                    d["room"].Tint = self.Tfloor # if floor, just use floor temp
                else:
                    Evt = d["vent"].timeStep(self.t, Tint = d["room"].Tint, Tout = self.Tout[c])
                    d["Vnvs"][c] = d["vent"].Vnv
                    d["room"].timeStep(d["Ef"], Evt)
                d["Tints"][c] = d["room"].Tint
                d["Ef"] = 0 #resetting Ef for next time step


class WallFlux:
    def __init__(self, front = None, back = None):
        self.front = front
        self.back = back

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
        expected_kwards = set(["X", "Y"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        # Constants
        self.rhof = 2300 #density of fabric
        self.Cf = 750 #specific heat capacity of fabric
        self.kf = 0.8 #thermal conductivity of fabric
        self.Af = self.X * self.Y #fabric area
        self.th = 0.10 #fabric thickness 
        self.h = WallFlux(4, 4) #fabric convection coefficient
        self.delx = 0.010 #spatial discretization size
        self.x = np.arange(0, self.th + self.delx, self.delx)
        self.alpha = 0.7 #fabric absorptivity

    def initialize(self, delt, TfF, TfB):
        # Scaling factors
        self.lambda_val = (self.kf * delt) / (self.rhof * self.Cf * self.delx**2)
        self.lambda_bound_F = self.kf / (self.h.front * self.delx)
        self.lambda_bound_B = self.kf / (self.h.back * self.delx)

        # Wall setup
        self.n = round(self.th / self.delx) - 1
        r = np.zeros(self.n)
        r[0] = 1 - 2 * self.lambda_val
        r[1] = self.lambda_val
        r[-1] = self.lambda_val

        A_matrix = sp_linalg.toeplitz(r)
        A_matrix[0, 0] += self.lambda_val * self.lambda_bound_F / (1 + self.lambda_bound_F)
        A_matrix[-1, -1] += self.lambda_val * self.lambda_bound_B / (1 + self.lambda_bound_B)
        A_matrix[-1, 0] = 0
        A_matrix[0, -1] = 0

        self.A = A_matrix

        self.b = np.zeros(self.n)
        self.T_prof = np.linspace(TfF, TfB, self.n + 2) #create a uniform temperature profile between Tff and Tfb of length n
        self.T = self.T_prof[1:-1] #remove the boundary temperatures from the temperature profile

        self.qradF = 0 #radiative heat flux at front
        self.qradB = 0 #radiative heat flux at back

    def timeStep(self, TintF, TintB):
        TintRadF = TintF + self.qradF / self.h.front
        TintRadB = TintB + self.qradB / self.h.back
        self.b[0] = self.lambda_val * TintRadF / (1 + self.lambda_bound_F)
        self.b[-1] = self.lambda_val * TintRadB / (1 + self.lambda_bound_B)
        self.T = np.dot(self.A, self.T) + self.b
        # self.T = np.linalg.solve(self.A, self.b)
        self.T_prof = self.getWallProfile(TintRadF, TintRadB)

        Ef = WallFlux()
        # Ef.front = self.Af * (self.T_prof[1] - self.T_prof[0]) / self.delx
        # Ef.back = self.Af * (self.T_prof[-2] - self.T_prof[-1]) / self.delx
        Ef.front = self.Af * (self.T_prof[0] - TintF) * self.h.front
        Ef.back = self.Af * (self.T_prof[-1] - TintB) * self.h.back
        return Ef

    def getWallProfile(self, TintF, TintB):
        T_prof = np.zeros(self.n + 2)
        T_prof[1:-1] = self.T
        T_prof[0] = self.get_Tf(self.T[0], TintF, self.lambda_bound_F)
        T_prof[-1] = self.get_Tf(self.T[-1], TintB, self.lambda_bound_B)
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
        if self.ventType == None:
            return 0
    

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

    def updateEdges(self, properties: dict, nodes = None):
        for i, j, d in self.G.edges(data=True):
            if nodes is None or i in nodes or j in nodes:
                d.update(deepcopy(properties))

    def updateNodes(self, properties: dict, nodes = None):
        for n, d in self.G.nodes(data=True):
            if nodes is None or n in nodes:
                d.update(deepcopy(properties))

class Radiation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["bG"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        
        # Constants
        self.sigma = 5.67e-8

    def initialize(self, roomNode:nx.classes.coreviews.AtlasView):
        self.roomNode = roomNode
        # assign properties to raidation graph
        for n, d in self.bG.G.nodes(data=True):
            if n == "sun":
                epislon = 1
                A = 1 # doesn't matter sice epsilon = 1
            else:
                if n == self.roomNode[n]["front"]:
                    d["T_index"] = -1 # reversed because front is in other room
                elif n == self.roomNode[n]["back"]:
                    d["T_index"] = 0 #reversed because back is in other room
                else:
                    raise Exception("Wall front/back has been missassigned")
                
                wall = self.roomNode[n]["wall"]
                epislon = wall.alpha # opaque, diffuse, gray surface
                d["X"] = wall.X  
                d["Y"] = wall.Y
                A = d["X"] * d["Y"]
            d["boundaryResistance"] = (1 - epislon) / (epislon * A)

        for i, j, d in self.bG.G.edges(data=True):
            #don't use properties of "sun" node
            if i == "sun":
                i = j
            elif j == "sun":
                j = i
            #calc radiance resistance
            X = self.bG.G.nodes[i]["X"]
            Y = self.bG.G.nodes[i]["Y"]
            if X != self.bG.G.nodes[j]["X"] or Y != self.bG.G.nodes[j]["Y"]:
                raise Exception("Dimmensions of adjacent nodes must be equal")
            A = X * Y
            F = getVFAlignedRectangles(X, Y, d["weight"])
            d["radianceResistance"] = (A * F) ** -1
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
            bR[n] = d["boundaryResistance"]
        J = np.linalg.solve(self.A, Eb)
        J = pd.Series(J, index = self.A.index)
        q = (J - Eb) / bR
        return  q # radiative gain for each node