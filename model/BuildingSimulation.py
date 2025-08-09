import numpy as np
from tqdm import tqdm
from model.utils import *
from model import \
    RoomSimulation as rs, \
    VentilationSimulation as vs, \
    WallSimulation as ws, \
    Radiation as rd, \
    BuildingGraph as bg


class BuildingSimulation():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["delt", "simLength", "Tout", "hradG", "vradG", "Tfloor"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        self.t = 0 #time (seconds)
        self.hour = 0 #time (hours)
        self.times = np.arange(0, self.simLength + self.delt, self.delt)
        self.hours = self.times / 60 / 60
        self.N = len(self.times)
        self.Tout = getEquivalentTimeSeries(self.Tout, self.times)
        self.hradG = getEquivalentTimeSeries(self.hradG, self.times)
        self.vradG = getEquivalentTimeSeries(self.vradG, self.times)
        self.radDamping =  self.delt / (1 + self.delt)# 0 damping factor for radiation

    def initialize(self, bG:bg.BuildingGraph, verbose = False):
        self.bG = bG
        for n, d in self.bG.G.nodes(data=True):
            r = rs.RoomSimulation(**d["room_kwargs"])
            v = vs.VentilationSimulation(**d["vent_kwargs"])
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
            d["wall_kwargs"]["delt"] = self.delt
            w = ws.WallSimulation(**d["wall_kwargs"]) # instantiate wall
            Tff = self.bG.G.nodes[d["nodes"].front]["room"].Tint #set wall front fabric temp    
            Tfb = self.bG.G.nodes[d["nodes"].back]["room"].Tint # set wall back fabric temp
            # set arbitrarily large convective heat transfer coefficient for floor-to-ground interface
            if d["nodes"].front == "FL":
                w.h.front = 1e6
            elif d["nodes"].back == "FL":
                w.h.back = 1e6
            w.initialize(self.delt, Tff, Tfb, verbose=verbose) #initialize wall

            T_profs = np.zeros((w.n + 2, self.N)) # intializing matrix to store temperature profiles
            T_profs[:, 0] = w.getWallProfile(Tff, Tfb) # store initial temperature profile

            radEApplied = WallSides() # initialize applied radiation as wall-side object
            radECalc = WallSides() # initialize calculated radiation as wall-side object
            for radE in [radEApplied, radECalc]:
                radE.front = np.zeros(self.N)
                radE.back = np.zeros(self.N)

            # store wall and related data as edge properties in graph 
            d.update({
                "wall": w,
                "T_profs": T_profs,
                "radEApplied": radEApplied,
                "radECalc": radECalc,
                })
        for n, d in self.bG.G.nodes(data=True):
            rad = rd.Radiation(**d["rad_kwargs"])
            if verbose:
                print(f"Initializing radiation for {n}")
            rad.initialize(self.bG.G[n])
            d.update({"rad": rad})


    def run(self):
        print(f"Running simulation for {self.N - 1} time steps")
        for c in tqdm(range(1, self.N), desc="Time Steps"):
            self.t = self.times[c]
            self.hour = self.t / 60 / 60

            # Simulation logic
            # Solve Radiation
            for n, d in self.bG.G.nodes(data=True):
                E = d["rad"].timeStep(solarGain = self.hradG[c])
                E = E.dropna()
                for wall, EWall in E.items():
                    if wall == "sky":
                        continue
                    self.bG.G.edges[n, wall]["nodes"].checkSides(n) # only for error checking
                    if n == self.bG.G.edges[n, wall]["nodes"].front:
                        radECalc = self.bG.G.edges[(n, wall)]["radECalc"].front
                        radEApplied = self.bG.G.edges[(n, wall)]["radEApplied"].front
                        self.bG.G.edges[(n, wall)]["wall"].Erad.setUpdateFront()
                    if n == self.bG.G.edges[n, wall]["nodes"].back:
                        radECalc = self.bG.G.edges[(n, wall)]["radECalc"].back
                        radEApplied = self.bG.G.edges[(n, wall)]["radEApplied"].back
                        self.bG.G.edges[(n, wall)]["wall"].Erad.setUpdateBack()
                    radECalc[c] = EWall # leveraging pass by assignment here
                    Ewall = (1 - self.radDamping) * EWall + self.radDamping * radEApplied[c - 1]
                    radEApplied[c] = Ewall
                    self.bG.G.edges[(n, wall)]["wall"].Erad.update(Ewall)
                    self.bG.G.edges[(n, wall)]["wall"].Erad.back = self.bG.G.edges[(n, wall)]["radEApplied"].back[c]

            # Solve Walls
            for i, j, d in self.bG.G.edges(data=True):
                Ef = d["wall"].timeStep(self.bG.G.nodes[d["nodes"].front]["room"].Tint, self.bG.G.nodes[d["nodes"].back]["room"].Tint)
                self.bG.G.nodes[d["nodes"].front]["Ef"] += Ef.front * d["weight"]
                self.bG.G.nodes[d["nodes"].back]["Ef"] += Ef.back * d["weight"]
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