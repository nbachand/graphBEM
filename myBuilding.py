import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from model import BuildingSimulation as bs, BuildingGraph as bg
from model.utils import *
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
from epw import epw
import plotly.express as px


def runMyBEM(
        weather_data,
        materials,
        makePlots = True):
    wallMaterial = materials["wall"]
    partitionMaterial = materials["partition"]
    roofMaterial = materials["roof"]
    floorMaterial = materials["floor"]
    times = weather_data.index.to_series().apply(lambda x: x.timestamp())
    times -= times[0]
    Touts = weather_data["Dry Bulb Temperature"].values + 273.15
    rad = weather_data["Total Sky Radiation"].values

    # Plotting the weather data
    if makePlots:
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(times, Touts, label='Temperature (°K)')
        plt.title('Daily Temperature Variation')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Temperature (°C)')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(times, rad, label='Radiation (W/m^2)', color='orange')
        plt.title('Daily Radiation Variation')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Radiation (W/m^2)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Specify Graph
    wW = 1- .25**2 #window-wall area
    dW = 1 - .25*.75 #door-wall area
    interiorRooms = ["CR", "SS", "DR", "CV"]
    roomList = [*interiorRooms, "OD", "RF", "FL"]

    connectivityMatrix = np.array([
        [0, 1, 1, 0, 2*wW, 1, 1],
        [1, 0, 1 ,1, wW, 1, 1],
        [1, 1, dW, 1, 3*wW, 2, 2],
        [0, 1, 1, 0, 2*wW + 2, 1, 1],
    ])

    bG = bg.BuildingGraph(connectivityMatrix, roomList)
    if makePlots:
        bG.draw()

    # Window dimmensions
    H = 1
    W = 1

    alphas = []
    As = []
    Ls = []

    sim_kwargs = {
        "delt": times[1] - times[0],
        "simLength": times[-1] - times[0],
        "Tout" : Touts,
        "radG": rad,
        "Tfloor": np.mean(Touts) - 2.5,
    }
    wall_kwargs = {"X": 4, "Y": 3, "material_df": partitionMaterial}
    wall_kwargs_OD = {"X": 4, "Y": 3, "material_df": wallMaterial}
    wall_kwargs_RF = {"X": 4, "Y": 4, "material_df": roofMaterial}
    wall_kwargs_FL = {"X": 4, "Y": 4, "material_df": floorMaterial}

    room_kwargs = {
        "T0": Touts[0],
        "V" : 4**2 * 3, #volume of air
        "Eint" : 0 #internal heat generation
    }
    vent_kwargs = {
        'H': 1,
        'W' : 1,
        "ventType": None,
        "alphas": alphas,
        "As": As,
        "Ls": Ls,
    }
    rad_kwargs_RF = {
        "bG": bg.BuildingGraph(
            np.array([
                [0, -1, -1, -1, -1],
                [-1, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
            ]),
            ["sky", "SS", "DR", "CV",  "CR", ]
        ),
    }

    rad_kwargs_FL = {
        "bG": bg.BuildingGraph(
            np.array([
                [0, 3],
                [3, 0],
            ]),
            ["RF", "FL"]
        ),
    }

    bG.updateEdges({"wall_kwargs" :wall_kwargs})
    bG.updateEdges({"wall_kwargs" :wall_kwargs_OD}, nodes=["OD"])
    bG.updateEdges({"wall_kwargs" :wall_kwargs_RF}, nodes=["RF"])
    bG.updateEdges({"wall_kwargs" :wall_kwargs_FL}, nodes=["FL"])
    for e in [("CV", "RF"), ("CV", "FL")]:
        bG.G.edges[e]["wall_kwargs"]["X"] *= 2

    bG.updateNodes({
        "room_kwargs": room_kwargs,
        "vent_kwargs": vent_kwargs,
        "rad_kwargs": {"bG": bg.BuildingGraph()},
        })
    bG.updateNodes({"rad_kwargs": rad_kwargs_RF}, nodes=["RF"])
    bG.updateNodes({"rad_kwargs": rad_kwargs_FL}, nodes=["SS", "DR", "CV", "CR"])

    for r in ["CR", "DR"]:
        bG.G.nodes[r]["room_kwargs"]["V"] *= 2

    build_sim = bs.BuildingSimulation(**sim_kwargs)
    build_sim.initialize(bG)
    build_sim.run()

    return build_sim, interiorRooms