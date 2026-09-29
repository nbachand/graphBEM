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
        expected_kwards = set(["delt", "simLength", "Tbound", "windSpeed", "radG"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        self.t = 0 #time (seconds)
        self.hour = 0 #time (hours)
        self.times = np.arange(0, self.simLength + self.delt, self.delt)
        self.hours = self.times / 60 / 60
        self.N = len(self.times)
        for node in self.Tbound:
            self.Tbound[node] = getEquivalentTimeSeries(self.Tbound[node], self.times)
        for node in self.windSpeed:
            self.windSpeed[node] = getEquivalentTimeSeries(self.windSpeed[node], self.times)
        for node in self.radG:
            forcing = self.radG[node]
            if isinstance(forcing, dict):
                if set(forcing) != {"shortwave", "longwave"}:
                    raise ValueError("Radiation forcing needs shortwave and longwave bands")
                self.radG[node] = {band: getEquivalentTimeSeries(values, self.times)
                                   for band, values in forcing.items()}
            else:
                # Legacy scalar/array inputs denote incident shortwave only.
                self.radG[node] = getEquivalentTimeSeries(forcing, self.times)
        self.radDamping =  self.delt / (1 + self.delt)# 0 damping factor for radiation

    def initialize(self, bG:bg.BuildingGraph, verbose = False):
        self.bG = bG
        for n, d in self.bG.G.nodes(data=True):
            r = rs.RoomSimulation(**d["room_kwargs"])
            v = vs.VentilationSimulation(**d["vent_kwargs"])
            r.initialize(self.delt)
            if n in self.Tbound:
                r.Tint = self.Tbound[n][0]

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
            hCalced = WallSides()
            for quantity in [radEApplied, radECalc, hCalced]:
                quantity.front = np.zeros(self.N)
                quantity.back = np.zeros(self.N)

            # store wall and related data as edge properties in graph 
            d.update({
                "wall": w,
                "T_profs": T_profs,
                "radEApplied": radEApplied,
                "radECalc": radECalc,
                "hCalced": hCalced,
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
                if n in self.radG:
                    forcing = self.radG[n]
                    if isinstance(forcing, dict):
                        d["rad"].solarGain = forcing["shortwave"][c]
                        d["rad"].longwaveGain = forcing["longwave"][c]
                    else:
                        d["rad"].solarGain = forcing[c]
                        d["rad"].longwaveGain = 0
                else:
                    d["rad"].solarGain = 0
                    d["rad"].longwaveGain = 0
                E = d["rad"].timeStep()
                E = E.dropna()
                for wall, EWall in E.items():
                    if wall == "sky":
                        continue
                    edge = self.bG.G.edges[n, wall]
                    edge["nodes"].checkSides(n)
                    # A self-loop partition represents two faces in this room.
                    # EWall is per unit face area, so apply it to each face.
                    for side in ("front", "back"):
                        if getattr(edge["nodes"], side) != n:
                            continue
                        calculated = getattr(edge["radECalc"], side)
                        applied = getattr(edge["radEApplied"], side)
                        calculated[c] = EWall
                        applied[c] = ((1 - self.radDamping) * EWall
                                      + self.radDamping * applied[c - 1])
                        setattr(edge["wall"].Erad, side, applied[c])

            # Solve Walls
            for i, j, d in self.bG.G.edges(data=True):
                if i in self.windSpeed:
                    d["wall"].windSpeed = self.windSpeed[i][c]
                elif j in self.windSpeed:
                    d["wall"].windSpeed = self.windSpeed[j][c]
                else:
                    d["wall"].windSpeed = 0
                Ef = d["wall"].timeStep(self.bG.G.nodes[d["nodes"].front]["room"].Tint, self.bG.G.nodes[d["nodes"].back]["room"].Tint)
                self.bG.G.nodes[d["nodes"].front]["Ef"] += Ef.front * d["weight"]
                self.bG.G.nodes[d["nodes"].back]["Ef"] += Ef.back * d["weight"]
                d["T_profs"][:,c] = d["wall"].T_prof
                d["hCalced"].front[c] = d["wall"].hCalced.front
                d["hCalced"].back[c] = d["wall"].hCalced.back

            # Solve Rooms
            for n, d in self.bG.G.nodes(data=True):
                if n in self.Tbound:
                    d["room"].Tint = self.Tbound[n][c] # assign boundary temp
                else:
                    Tout = self.Tbound["OD"][c]
                    Evt = d["vent"].timeStep(self.t, Tint = d["room"].Tint, Tout = Tout)
                    d["Vnvs"][c] = d["vent"].Vnv
                    d["room"].timeStep(d["Ef"], Evt)
                d["Tints"][c] = d["room"].Tint
                d["Ef"] = 0 #resetting Ef for next time step