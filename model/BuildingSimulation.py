import numpy as np
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
        expected_kwards = set(["delt", "simLength", "Tout", "radG", "Tfloor"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        self.t = 0 #time (seconds)
        self.hour = 0 #time (hours)
        self.times = np.arange(0, self.simLength + self.delt, self.delt)
        self.hours = self.times / 60 / 60
        self.N = len(self.times)
        self.Tout = getEquivalentTimeSeries(self.Tout, self.times)
        self.radG = getEquivalentTimeSeries(self.radG, self.times)
        self.radDamping = 0 #self.delt / (1 + self.delt)# damping factor for radiation

    def initialize(self, bG:bg.BuildingGraph):
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
            w = ws.WallSimulation(**d["wall_kwargs"])
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
            rad = rd.Radiation(**d["rad_kwargs"])
            rad.initialize(self.bG.G[n])
            d.update({"rad": rad})


    def run(self):
        for c in range(1, self.N):
            self.t = self.times[c]
            self.hour = self.t / 60 / 60

            # Simulation logic
            # Solve Radiation
            for n, d in self.bG.G.nodes(data=True):
                E = d["rad"].timeStep(self.radG[c])
                E = E.dropna()
                for wall, EWall in E.items():
                    if wall == "sun":
                        continue
                    if n == self.bG.G.edges[n, wall]["front"]:
                        EOld = self.bG.G.edges[wall, n]["wall"].EradF
                        self.bG.G.edges[(n, wall)]["radECalc"].front[c] = EWall
                        EWall = (1 - self.radDamping) * EWall + self.radDamping * EOld
                        self.bG.G.edges[(n, wall)]["radEApplied"].front[c] = EWall
                        self.bG.G.edges[wall, n]["wall"].EradF = EWall
                    elif n == self.bG.G.edges[n, wall]["back"]:
                        EOld = self.bG.G.edges[wall, n]["wall"].EradB
                        self.bG.G.edges[(n, wall)]["radECalc"].back[c] = EWall
                        EWall = (1 - self.radDamping) * EWall + self.radDamping * EOld
                        self.bG.G.edges[(n, wall)]["radEApplied"].back[c] = EWall
                        self.bG.G.edges[wall, n]["wall"].EradB = EWall
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