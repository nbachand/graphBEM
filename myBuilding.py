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

def tempPlotBasics():
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=-45)
    plt.xlabel('Time')
    plt.ylabel('Temperature [K]')
    plt.tight_layout()
    return

def plotVentLines(t, allVent):
    # if allVent == False:
    plt.axvline(t, linestyle = '-.', color = '.8')
    return

def runMyBEM(
        weather_data,
        materials,
        floorTempAdjustment,
        hInterior,
        hExterior,
        alphaRoof,
        allVent = False,
        startVentHour = 16,
        otherVentHours = [23],
        coolingThreshold = 273.15 + 24, # 24 C or 75 F
        verbose = False,
        makePlots = False):
    
    outputs = {}
    wallMaterial = materials["wall"]
    partitionMaterial =  materials["partition"]
    roofMaterial = materials["roof"]
    floorMaterial = materials["floor"]
    if verbose:
        print("floor material:")
        print(floorMaterial)
        print("wall material:")
        print(wallMaterial)
        print("partition material:")
        print(partitionMaterial)
        print("roof material:")
        print(roofMaterial)
    times = weather_data.index.to_series().apply(lambda x: x.timestamp())
    times -= times.iloc[0]
    dt = times.iloc[1]
    Touts = weather_data["Dry Bulb Temperature"].values + 273.15
    rad = weather_data["Total Sky Radiation"].values

    # Plotting the weather data
    if makePlots:
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(times.index.values, Touts, label='Temperature (°K)')
        plt.title('Daily Temperature Variation')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.subplot(2, 1, 2)
        plt.plot(times.index.values, rad, label='Radiation (W/m^2)', color='orange')
        plt.title('Daily Radiation Variation')
        plt.xlabel('Time')
        plt.ylabel('Radiation (W/m^2)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

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
        "Tfloor": np.mean(Touts) + floorTempAdjustment,
    }
    wall_kwargs = {"X": 4, "Y": 3, "material_df": partitionMaterial, "h": WallSides(hInterior, hInterior), "alpha" : 0.7}
    wall_kwargs_OD = {"X": 4, "Y": 3, "material_df": wallMaterial,   "h": WallSides(hInterior, hExterior), "alpha" : 0.7}
    wall_kwargs_RF = {"X": 4, "Y": 4, "material_df": roofMaterial,   "h": WallSides(hInterior, hExterior), "alpha" : alphaRoof}
    wall_kwargs_FL = {"X": 4, "Y": 4, "material_df": floorMaterial,  "h": WallSides(hInterior, 1e6), "alpha" : 0.7}

    room_kwargs = {
        "T0": np.mean(Touts), #Touts[0],
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
    build_sim.initialize(bG, verbose=verbose)
    build_sim.run()

    #### Ventilation Times:

    Tints_avg = []
    if makePlots:
        plt.figure(figsize=(10, 6))
    for n, d in build_sim.bG.G.nodes(data=True):
        if n in interiorRooms:
            Tints_avg.append(d['Tints'])
            if makePlots:
                plt.plot(times.index.values, d['Tints'], label=n)
    Tints_avg = np.mean(np.array(Tints_avg), axis = 0)

    Tout_minus_in = build_sim.Tout - Tints_avg

    hVent = []
    iVent = []
    outputs["dVent"] = []
    outputs["nVent"] = []
    outputs["hVent"] = []
    outputs["Tint"] = []
    outputs["Tout"] = []
    outputs["ToutMinusTint"] = []
    outputs["maxToutVent"] = []
    T_old = 0
    stepsDay = 24 * 60 * 60 / dt
    iVentMin = stepsDay / 4
    n = 0 # tracking the nth ventilation time of the night
    lastMaxTout = 0
    for i, T in enumerate(Tout_minus_in):
        h = times.index.hour[i]
        if h == 0:
            lastMaxTout = 0
        elif build_sim.Tout[i] > lastMaxTout:
            lastMaxTout = build_sim.Tout[i]
        if T <= 0 and i > iVentMin and Tints_avg[i] > coolingThreshold and h > startVentHour and (T_old > 0 or allVent == True or h in otherVentHours):
            if i > iVentMin + 1: # indicating this is not a continuing ventilation
                day = times.index.day[i] - times.index.day[0]
                lastMaxToutVent = lastMaxTout #making sure this is not reset during the building ventilation period
                if h not in otherVentHours:
                    n = 0
            if allVent == True or len(otherVentHours) > 0:
                iVentMin = i + stepsDay / 24
            else: 
                iVentMin = i + stepsDay / 2 # Wait at least half a day before venting again
            hVent.append(times.index.hour[i])
            iVent.append(i)
            outputs["dVent"].append(day)
            outputs["nVent"].append(n)
            outputs["hVent"].append(h)
            outputs["Tint"].append(Tints_avg[i])
            outputs["Tout"].append(build_sim.Tout[i])
            outputs["ToutMinusTint"].append(T)
            outputs["maxToutVent"].append(lastMaxToutVent)
            n += 1
            if verbose and allVent == False:
                print(f"Ventilation at {round(hVent[-1],1)} hours (time: {round(hVent[-1]%24, 1)})")
        if Tints_avg[i] < coolingThreshold:
            T_old = 1 # waiting for indoor temperature to get to hot
        else:
            T_old = T

    outputs["hVent"] = hVent

    #### General Plots

    if makePlots:
        for i in iVent:
            plotVentLines(times.index.values[i], allVent)
        plt.plot(times.index.values, Tints_avg, label="Average Interior Temperature", color = 'k', linestyle = '--')
        plt.plot(times.index.values, build_sim.Tout, label="Outdoor Temperature", color = 'k', linestyle = (0, (1, 5)))
        tempPlotBasics()
        plt.title("Room Temperatures")

        plt.figure(figsize=(10, 6))
        for i in iVent:
            plotVentLines(times.index.values[i], allVent)
        c = 0
        for i, j, d, in build_sim.bG.G.edges(data=True):
            if d['nodes'].checkSides(i, False) in interiorRooms and d['nodes'].checkSides(j, False) in interiorRooms:
                colors = list(mcolors.TABLEAU_COLORS.keys())
                linetypes = ['-', '--']
                plt.plot(times.index.values, d['T_profs'][0, :], label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[0])
                plt.plot(times.index.values, d['T_profs'][-1, :], label=f'{i}-{j}-B', color = colors[c], linestyle = linetypes[1])
                c = (c + 1) % len(colors)
        plt.plot(times.index.values, build_sim.Tout, label="Outdoor Temperature", color = 'k', linestyle = (0, (1, 5)))
        tempPlotBasics()
        plt.title("Interior Wall Suface Temperatures")

        plt.figure(figsize=(10, 6))
        for i in iVent:
            plotVentLines(times.index.values[i], allVent)
        c = 0
        for i, j, d, in build_sim.bG.G.edges(data=True):
            if d['nodes'].checkSides(i, False) == "OD" or d['nodes'].checkSides(j, False) == "OD":
                plt.plot(times.index.values, d['T_profs'][0, :], label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[0])
                plt.plot(times.index.values, d['T_profs'][-1, :], label=f'{i}-{j}-B', color = colors[c], linestyle = linetypes[1])
                c = (c + 1) % len(colors)
        plt.plot(times.index.values, build_sim.Tout, label="Outdoor Temperature", color = 'k', linestyle = (0, (1, 5)))
        tempPlotBasics()
        plt.title("Exterior Wall Suface Temperatures")

        plt.figure(figsize=(10, 6))
        for i in iVent:
            plotVentLines(times.index.values[i], allVent)
        c = 0
        for i, j, d, in build_sim.bG.G.edges(data=True):
            if d['nodes'].checkSides(i, False) == "RF" or d['nodes'].checkSides(j, False) == "RF":
                plt.plot(times.index.values, d['T_profs'][0, :], label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[0])
                plt.plot(times.index.values, d['T_profs'][-1, :], label=f'{i}-{j}-B', color = colors[c], linestyle = linetypes[1])
                c = (c + 1) % len(colors)
        plt.plot(times.index.values, build_sim.Tout, label="Outdoor Temperature", color = 'k', linestyle = (0, (1, 5)))
        tempPlotBasics()
        plt.title("Roof Suface Temperatures")

        plt.figure(figsize=(10, 6))
        for i in iVent:
            plotVentLines(times.index.values[i], allVent)
        c = 0
        for i, j, d, in build_sim.bG.G.edges(data=True):
            if d['nodes'].checkSides(i, False) == "RF" or d['nodes'].checkSides(j, False) == "RF":
                plt.plot(times.index.values, d['radECalc'].back, label=f'{i}-{j}-C', color = colors[c], linestyle = linetypes[0])
                plt.plot(times.index.values, d['radEApplied'].back, label=f'{i}-{j}-A', color = colors[c], linestyle = linetypes[1])
                plt.plot(times.index.values, d['radECalc'].front, color = colors[c], linestyle = linetypes[0])
                plt.plot(times.index.values, d['radEApplied'].front, color = colors[c], linestyle = linetypes[1])
                c = (c + 1) % len(colors)
        plt.plot(times.index.values, build_sim.radG, label="Solar Radiation", color = 'k', linestyle = (0, (1, 5)))
        tempPlotBasics()
        plt.ylabel('Energy Flux [W/m^2]')
        plt.title("Roof Radiative Fluxes")

        plt.figure(figsize=(10, 6))
        for i in iVent:
            plotVentLines(times.index.values[i], allVent)
        c = 0
        for i, j, d, in build_sim.bG.G.edges(data=True):
            if d['nodes'].checkSides(i, False) == "FL" or d['nodes'].checkSides(j, False) == "FL":
                plt.plot(times.index.values, d['T_profs'][0, :], label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[0])
                plt.plot(times.index.values, d['T_profs'][-1, :], label=f'{i}-{j}-B', color = colors[c], linestyle = linetypes[1])
                c = (c + 1) % len(colors)
        plt.plot(times.index.values, build_sim.Tout, label="Outdoor Temperature", color = 'k', linestyle = (0, (1, 5)))
        tempPlotBasics()
        plt.title("Floor Suface Temperatures")

        fig, axs = plt.subplots(3, 1, figsize=(8, 18))
        for i in iVent:
            for j in range(3):
                if allVent == False:
                    axs[j].axvline(times.index.values[i], linestyle = '-.', color = '.8')
                    axs[j].plot(times.index.values, build_sim.Tout, label="Outdoor Temperature", color = 'k', linestyle = (0, (1, 5)))
        for i, j, d in build_sim.bG.G.edges(data=True):
            center = int(len(d['T_profs'][:, 0]) / 2)
            if d['nodes'].checkSides(i, False) in interiorRooms and d['nodes'].checkSides(j, False) in interiorRooms:
                lt = '-'
                label = "interior wall"
            elif d['nodes'].checkSides(i, False) == "OD" or d['nodes'].checkSides(j, False) == "OD":
                lt = '--'
                label = "exterior wall"
            elif d['nodes'].checkSides(i, False) == "RF" or d['nodes'].checkSides(j, False) == "RF":
                lt = '-.'
                label = "roof"
            elif d['nodes'].checkSides(i, False) == "FL" or d['nodes'].checkSides(j, False) == "FL":
                lt = ':'
                label = "floor"
            axs[0].plot(times.index.values, d['T_profs'][0, :], color = 'k', linestyle = lt, label = label)
            axs[1].plot(times.index.values, d['T_profs'][center, :], color = 'k', linestyle = lt, label = label)
            axs[2].plot(times.index.values, d['T_profs'][-1, :], color = 'k', linestyle = lt, label = label)
            # plt.legend()

        # Get handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Create a dictionary to store unique labels and their corresponding handles
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle

        # Plot with unique labels
        for i in range(3):
            axs[i].legend(unique_labels.values(), unique_labels.keys())

        axs[0].set_title("front")
        axs[1].set_title("center")
        axs[2].set_title("back")
        plt.tight_layout()

        #### Wall profiles
        plt.figure(figsize=(10, 6))
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

        plt.figure(figsize=(10, 6))
        wall = build_sim.bG.G.edges['DR', 'FL']['wall']
        T_profs = build_sim.bG.G.edges['DR', 'FL']['T_profs']
        for h in h_profs:
            i = int((24 - h) * 60 * 60 / build_sim.delt)
            plt.plot(wall.x, T_profs[:, i], label=f'{h % 24} hours')

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Temperature [K]')
        plt.show()

        plt.figure(figsize=(10, 6))
        wall = build_sim.bG.G.edges['DR', 'RF']['wall']
        T_profs = build_sim.bG.G.edges['DR', 'RF']['T_profs']
        for h in h_profs:
            i = int((24 - h) * 60 * 60 / build_sim.delt)
            plt.plot(wall.x, T_profs[:, i], label=f'{h % 24} hours')

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Temperature [K]')
        plt.show()


    if makePlots:
        plt.figure(figsize=(10, 6))
        for i in iVent:
            plotVentLines(times.index.values[i], allVent)
        plt.plot(times.index.values, Tout_minus_in, label="Outdoor-Indoor temperature difference")
        plt.plot(times.index.values, np.zeros_like(times.index.values), label="Indoor temperature")
        # plt.scatter(hVent, np.zeros_like(hVent), label="Ventilation Times")

    #### Temperature Differences
    if makePlots:
        plt.figure(figsize=(10, 6))
        for i in iVent:
            plotVentLines(times.index.values[i], allVent)
        plt.plot(times.index.values, Tout_minus_in, label="Outdoor-Indoor Temperature Difference", color = 'k', linestyle = '--')
        for i in iVent:
            plotVentLines(times.index.values[i], allVent)
        tempPlotBasics()
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
                plt.plot(times.index.values, diff, label=n)
                # plt.scatter(hVent, diff[iVent])
    outputs["ceilingMinusFloor"] = np.mean(delVent, axis = 0)
    outputs["outMinusFloor"] = np.mean(delOutFloor, axis = 0)
    if verbose:
        print(f'Average "ceiling - floor" temperature difference at ventilation time: {outputs["ceilingMinusFloor"]}')
        print(f'Average "outdoor - floor" temperature difference at ventilation time: {outputs["outMinusFloor"]}')
    
    if makePlots:
        plt.figure(figsize=(10, 6))
        for i in iVent:
            plotVentLines(times.index.values[i], allVent)
        plt.plot(times.index.values, Tout_minus_in, label="Outdoor-Indoor Temperature Difference", color = 'k', linestyle = '--')
        tempPlotBasics()
        plt.title("Interior Wall - Floor; Surface Temperatures")
        c = 0
    delVent = []
    for i, j, d, in build_sim.bG.G.edges(data=True):
        if d['nodes'].checkSides(i, False) in interiorRooms and d['nodes'].checkSides(j, False) in interiorRooms:
            colors = list(mcolors.TABLEAU_COLORS.keys())
            linetypes = ['-', '--']
            for side in range(2):
                floor_temp = build_sim.bG.G[[i, j][side]]["FL"]["T_profs"][0, :]
                diff = d['T_profs'][-side, :] - floor_temp
                delVent.append(diff[iVent])
                if makePlots:   
                    plt.plot(times.index.values, diff, label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[side])
                    # plt.scatter(hVent, diff[iVent], color = colors[c])
                    c = (c + 1) % len(colors)
    outputs["intWallMinusFloor"] = np.mean(delVent, axis = 0)
    if verbose:
        print(f'Average "interior wall - floor" temperature difference at ventilation time: {outputs["intWallMinusFloor"]}')

    if makePlots:
        plt.figure(figsize=(10, 6))
        for i in iVent:
            plotVentLines(times.index.values[i], allVent)
        plt.plot(times.index.values, Tout_minus_in, label="Outdoor-Indoor Temperature Difference", color = 'k', linestyle = '--')
        tempPlotBasics()
        plt.title("Exterior Wall - Floor; Surface Temperatures")
        c = 0
    delVent = []
    for i, j, d, in build_sim.bG.G.edges(data=True):
        if d['nodes'].checkSides(i, False) == "OD" or d['nodes'].checkSides(j, False) == "OD":
            colors = list(mcolors.TABLEAU_COLORS.keys())
            linetypes = ['-', '--']
            floor_temp = build_sim.bG.G[i]["FL"]["T_profs"][0, :]
            diff = d['T_profs'][0, :] - floor_temp
            delVent.append(diff[iVent])
            if makePlots:
                plt.plot(times.index.values, diff, label=f'{i}-{j}-F', color = colors[c], linestyle = linetypes[0])
                # plt.scatter(hVent, diff[iVent], color = colors[c])
                c = (c + 1) % len(colors)
    outputs["extWallMinusFloor"] = np.mean(delVent, axis = 0)
    if verbose:
        print(f'Average "exterior wall - floor" temperature difference at ventilation time: {outputs["extWallMinusFloor"]}')
    
    return outputs
