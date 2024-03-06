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
import dask

@dask.delayed
def runMyBEM(
        weather_data,
        materials,
        makePlots = True):
    
    outputs = {}
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
        "delt": times.values[1] - times.values[0],
        "simLength": times.values[-1] - times.values[0],
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
        "solveType": "sky"
    }

    rad_kwargs_FL = {
        "solveType": "room"
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
        "rad_kwargs": {"solveType": None},
        })
    bG.updateNodes({"rad_kwargs": rad_kwargs_RF}, nodes=["RF"])
    bG.updateNodes({"rad_kwargs": rad_kwargs_FL}, nodes=["SS", "DR", "CV", "CR"])

    for r in ["CR", "DR"]:
        bG.G.nodes[r]["room_kwargs"]["V"] *= 2

    build_sim = bs.BuildingSimulation(**sim_kwargs)
    build_sim.initialize(bG)
    build_sim.run()

    if makePlots:

        #### General Plots

        Tints_avg = []
        plt.figure()
        for n, d in build_sim.bG.G.nodes(data=True):
            if n in interiorRooms:
                plt.plot(build_sim.hours, d['Tints'], label=n)
                Tints_avg.append(d['Tints'])
        Tints_avg = np.mean(np.array(Tints_avg), axis = 0)
        plt.plot(build_sim.hours, Tints_avg, label="Average Interior Temperature", color = 'k', linestyle = '--')
        plt.plot(build_sim.hours, build_sim.Tout, label="Outdoor Temperature", color = 'k', linestyle = ':')
        plt.legend()
        plt.title("Room Temperatures")

        plt.figure()
        c = 0
        for i, j, d, in build_sim.bG.G.edges(data=True):
            if d['nodes'].checkSides(i, False) in interiorRooms and d['nodes'].checkSides(j, False) in interiorRooms:
                colors = list(mcolors.TABLEAU_COLORS.keys())
                linetypes = ['-', '--']
                plt.plot(build_sim.hours, d['T_profs'][0, :], label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[0])
                plt.plot(build_sim.hours, d['T_profs'][-1, :], label=f'{i}-{j}-B', color = colors[c], linestyle = linetypes[1])
                c = (c + 1) % len(colors)
        plt.plot(build_sim.hours, build_sim.Tout, label="Outdoor Temperature", color = 'k', linestyle = ':')
        plt.legend()
        plt.title("Interior Wall Surface Temperatures")

        plt.figure()
        c = 0
        for i, j, d, in build_sim.bG.G.edges(data=True):
            if d['nodes'].checkSides(i, False) == "OD" or d['nodes'].checkSides(j, False) == "OD":
                plt.plot(build_sim.hours, d['T_profs'][0, :], label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[0])
                plt.plot(build_sim.hours, d['T_profs'][-1, :], label=f'{i}-{j}-B', color = colors[c], linestyle = linetypes[1])
                c = (c + 1) % len(colors)
        plt.plot(build_sim.hours, build_sim.Tout, label="Outdoor Temperature", color = 'k', linestyle = ':')
        plt.legend()
        plt.title("Exterior Wall Suface Temperatures")

        plt.figure()
        c = 0
        for i, j, d, in build_sim.bG.G.edges(data=True):
            if d['nodes'].checkSides(i, False) == "RF" or d['nodes'].checkSides(j, False) == "RF":
                plt.plot(build_sim.hours, d['T_profs'][0, :], label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[0])
                plt.plot(build_sim.hours, d['T_profs'][-1, :], label=f'{i}-{j}-B', color = colors[c], linestyle = linetypes[1])
                c = (c + 1) % len(colors)
        plt.plot(build_sim.hours, build_sim.Tout, label="Outdoor Temperature", color = 'k', linestyle = ':')
        plt.legend()
        plt.title("Roof Suface Temperatures")

        plt.figure()
        c = 0
        for i, j, d, in build_sim.bG.G.edges(data=True):
            if d['nodes'].checkSides(i, False) == "FL" or d['nodes'].checkSides(j, False) == "FL":
                plt.plot(build_sim.hours, d['T_profs'][0, :], label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[0])
                plt.plot(build_sim.hours, d['T_profs'][-1, :], label=f'{i}-{j}-B', color = colors[c], linestyle = linetypes[1])
                c = (c + 1) % len(colors)
        plt.plot(build_sim.hours, build_sim.Tout, label="Outdoor Temperature", color = 'k', linestyle = ':')
        plt.legend()
        plt.title("Floor Suface Temperatures")

        plt.figure()
        for i, j, d in build_sim.bG.G.edges(data=True):
            center = int(len(d['T_profs'][:, 0]) / 2)
            if d['nodes'].checkSides(i, False) in interiorRooms and d['nodes'].checkSides(j, False) in interiorRooms:
                lt = '-'
            elif d['nodes'].checkSides(i, False) == "OD" or d['nodes'].checkSides(j, False) == "OD":
                lt = '--'
            elif d['nodes'].checkSides(i, False) == "RF" or d['nodes'].checkSides(j, False) == "RF":
                lt = '-.'
            elif d['nodes'].checkSides(i, False) == "FL" or d['nodes'].checkSides(j, False) == "FL":
                lt = ':'
            plt.plot(build_sim.hours, d['T_profs'][center, :], color = 'k', linestyle = lt)
            # plt.legend()
        plt.title("Wall Center Temperatures")

        plt.figure()
        for i, j, d in build_sim.bG.G.edges(data=True):
            center = int(len(d['T_profs'][:, 0]) / 2)
            if d['nodes'].checkSides(i, False) in interiorRooms and d['nodes'].checkSides(j, False) in interiorRooms:
                lt = '-'
                plt.plot(build_sim.hours, d['T_profs'][-1, :], color = 'k', linestyle = lt)
            elif d['nodes'].checkSides(i, False) == "OD" or d['nodes'].checkSides(j, False) == "OD":
                lt = '--'
            elif d['nodes'].checkSides(i, False) == "RF" or d['nodes'].checkSides(j, False) == "RF":
                lt = '-.'
            elif d['nodes'].checkSides(i, False) == "FL" or d['nodes'].checkSides(j, False) == "FL":
                lt = ':'
            plt.plot(build_sim.hours, d['T_profs'][0, :], color = 'k', linestyle = lt)
            # plt.legend()
        plt.title("Wall Center Temperatures")

        #### Wall profiles
        plt.figure()
        h_profs = [4, 8, 12, 16, 20]
        # h_profs = [h + 48 for h in h_profs]
        wall = build_sim.bG.G.edges['DR', 'DR']['wall']
        T_profs = build_sim.bG.G.edges['DR', 'DR']['T_profs']
        for h in h_profs:
            i = int((24 - h) * 60 * 60 / build_sim.delt)
            plt.plot(wall.x, T_profs[:, i], label=f'{h % 24} hours')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Temperature [K]')
        plt.show()

        plt.figure()
        wall = build_sim.bG.G.edges['DR', 'FL']['wall']
        T_profs = build_sim.bG.G.edges['DR', 'FL']['T_profs']
        for h in h_profs:
            i = int((24 - h) * 60 * 60 / build_sim.delt)
            plt.plot(wall.x, T_profs[:, i], label=f'{h % 24} hours')

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Temperature [K]')
        plt.show()

        plt.figure()
        wall = build_sim.bG.G.edges['DR', 'RF']['wall']
        T_profs = build_sim.bG.G.edges['DR', 'RF']['T_profs']
        for h in h_profs:
            i = int((24 - h) * 60 * 60 / build_sim.delt)
            plt.plot(wall.x, T_profs[:, i], label=f'{h % 24} hours')

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Temperature [K]')
        plt.show()


    #### Ventilation Times:
    Tout_minus_in = build_sim.Tout - Tints_avg

    # Tvent = 297
    hVent = []
    iVent = []
    # T_old = Tvent
    T_old = 0
    # for i, T in enumerate(build_sim.Tout):
        # if T_old > Tvent and T <= Tvent:
    for i, T in enumerate(Tout_minus_in):
        if T_old > 0 and T <= 0:
            hVent.append(build_sim.hours[i])
            iVent.append(i)
            print(f"Ventilation at {round(hVent[-1],1)} hours (time: {round(hVent[-1]%24, 1)})")
        T_old = T

    if makePlots:
        plt.figure()
        plt.plot(build_sim.hours, Tout_minus_in, label="Outdoor-Indoor temperature difference")
        plt.plot(build_sim.hours, np.zeros_like(build_sim.hours), label="Indoor temperature")
        plt.scatter(hVent, np.zeros_like(hVent), label="Ventilation Times")

    #### Temperature Differences
    if makePlots:
        plt.figure()
        plt.plot(build_sim.hours, Tout_minus_in, label="Outdoor-Indoor Temperature Difference", color = 'k', linestyle = '--')
        plt.legend()
        plt.title("Ceiling - Floor; Temperature Difference")
    delVent = []
    delOutFloor = []
    for n, _ in build_sim.bG.G.nodes(data=True):
        if n in interiorRooms:
            ceiling_temp = build_sim.bG.G[n]["RF"]["T_profs"][0, :]
            floor_temp = build_sim.bG.G[n]["FL"]["T_profs"][0, :]
            Tout_floor_diff = build_sim.Tout - floor_temp
            diff = ceiling_temp - floor_temp
            delVent.append(diff[iVent])
            delOutFloor.append(Tout_floor_diff[iVent])
            if makePlots:
                plt.plot(build_sim.hours, diff, label=n)
                plt.scatter(hVent, diff[iVent])
    outputs["ceilingMinusFloor"] = np.mean(delVent)
    outputs["outMinusFloor"] = np.mean(delOutFloor)
    display(f'Average "ceiling - floor" temperature difference at ventilation time: {outputs["ceilingMinusFloor"]}')
    display(f'Average "outdoor - floor" temperature difference at ventilation time: {outputs["outMinusFloor"]}')
    
    if makePlots:
        plt.figure()
        plt.plot(build_sim.hours, Tout_minus_in, label="Outdoor-Indoor Temperature Difference", color = 'k', linestyle = '--')
        plt.legend()
        plt.title("Interior Wall - Floor; Surface Temperatures")
        c = 0
    delVent = []
    for i, j, d, in build_sim.bG.G.edges(data=True):
        if d['nodes'].checkSides(i, False) in interiorRooms and d['nodes'].checkSides(j, False) in interiorRooms:
            colors = list(mcolors.TABLEAU_COLORS.keys())
            linetypes = ['-', '--']
            for side in range(2):
                floor_temp = build_sim.bG.G[[i, j][side]]["RF"]["T_profs"][0, :]
                diff = d['T_profs'][-side, :] - floor_temp
                delVent.append(diff[iVent])
                if makePlots:   
                    plt.plot(build_sim.hours, diff, label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[side])
                    plt.scatter(hVent, diff[iVent], color = colors[c])
                    c = (c + 1) % len(colors)
    outputs["intWallMinusFloor"] = np.mean(delVent)
    display(f'Average "interior wall - floor" temperature difference at ventilation time: {np.mean(delVent)}')

    if makePlots:
        plt.figure()
        plt.plot(build_sim.hours, Tout_minus_in, label="Outdoor-Indoor Temperature Difference", color = 'k', linestyle = '--')
        plt.legend()
        plt.title("Exterior Wall - Floor; Surface Temperatures")
        c = 0
    delVent = []
    for i, j, d, in build_sim.bG.G.edges(data=True):
        if d['nodes'].checkSides(i, False) == "OD" or d['nodes'].checkSides(j, False) == "OD":
            colors = list(mcolors.TABLEAU_COLORS.keys())
            linetypes = ['-', '--']
            floor_temp = build_sim.bG.G[i]["RF"]["T_profs"][0, :]
            diff = d['T_profs'][0, :] - floor_temp
            delVent.append(diff[iVent])
            if makePlots:
                plt.plot(build_sim.hours, diff, label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[0])
                plt.scatter(hVent, diff[iVent], color = colors[c])
                c = (c + 1) % len(colors)
    outputs["extWallMinusFloor"] = np.mean(delVent)
    display(f'Average "exterior wall - floor" temperature difference at ventilation time: {outputs["extWallMinusFloor"]}')
    
    return outputs