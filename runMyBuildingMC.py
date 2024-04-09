# %%

import numpy as np
from matplotlib import pyplot as plt
from runMC import runMC
from model.utils import *
from model.WallSimulation import convectionDOE2
from epw import epw
import random
import os
import time

plt.close('all')

def getConstructions(type, soil_depth = 0.5, soil_conductivity = 1.5, soil_density = 2800, soil_specific_heat = 850):
    floor = cleanMaterial(f"{type} Floor", reverse=False)
    floor.loc["Soil"] = ["Material", np.nan, soil_depth, soil_conductivity, soil_density, soil_specific_heat, np.nan]
    return {
        "wall": cleanMaterial(f"{type} Exterior Wall"),
        "partition": cleanMaterial(f"{type} Partitions"),
        "roof": cleanMaterial(f"{type} Roof/Ceiling"),
        "floor": floor
    }

# Function to generate sinusoidal data for temperature and radiation
def generate_sinusoidal_data(delt = 15, amplitude_temp=10, amplitude_radiation=250, length=24):
    length *= 3600  # Converting period from hours to seconds
    period = 24 * 3600
    time = np.arange(0, length, delt)  # Representing a day (24 hours) with a sinusoidal pattern
    temperature = amplitude_temp * np.sin(2 * np.pi * time / period) + 300  # Centered 
    radiation = amplitude_radiation * np.sin(2 * np.pi * time / period) + 250
    return time, temperature, radiation

def cleanMaterial(materialName, reverse = True):
    constructions  = pd.read_csv("energyPlus/ASHRAE_2005_HOF_Constructions.csv", index_col="Name")
    materials = pd.read_csv("energyPlus/ASHRAE_2005_HOF_Materials.csv", index_col="Name")
    wallLayers = ["Outside_Layer", "Layer_2", "Layer_3", "Layer_4", "Layer_5"]
    if reverse:
        wallLayers = wallLayers[::-1]
    material_df = []
    for layer in wallLayers:
        material = constructions[layer][materialName]
        if isinstance(material, str):
            conductivity = materials["Conductivity"][material]
            if (conductivity > 10) == False: # filter out the materials with conductivity > 10 that mess up solver and basically transfer all heat
                material_df.append(materials.loc[material])
    return pd.DataFrame(material_df)

def main(N = 100, runDays = 7, writeResults = True, randomSeed = 666, material_types = ["Light", "Medium", "Heavy"]):

    mainStart = time.time()

    # %% [markdown]
    #   # Specify Weather Data

    # %%

    # Generate sinusoidal data
    times, Touts, rad = generate_sinusoidal_data(delt=5, length=96)




    # %%
    a = epw()
    a.read("./energyPlus/weather/USA_CA_Sacramento.724835_TMY2.epw")
    a.dataframe.index = pd.to_datetime(a.dataframe[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    print(a.dataframe.keys())
    print(set(a.dataframe["Year"]))



    # %%
    data = a.dataframe[(a.dataframe["Month"]==8) & (a.dataframe["Day"] >= 1)]

    # fig, ax = plt.subplots()
    # ax.plot(data.index, data["Extraterrestrial Horizontal Radiation"], label = "Extraterrestrial Horizontal Radiation")
    # ax.plot(data.index, data["Extraterrestrial Direct Normal Radiation"], label = "Extraterrestrial Direct Normal Radiation")
    # ax.plot(data.index, data["Horizontal Infrared Radiation Intensity"], label = "Horizontal Infrared Radiation Intensity")
    # ax.plot(data.index, data["Global Horizontal Radiation"], label = "Global Horizontal Radiation")
    # ax.plot(data.index, data["Direct Normal Radiation"], label = "Direct Normal Radiation")
    # ax.plot(data.index, data["Diffuse Horizontal Radiation"], label = "Diffuse Horizontal Radiation")
    # ax.legend()

    data["Total Sky Radiation"] = data["Horizontal Infrared Radiation Intensity"] + data["Global Horizontal Radiation"]
    dt = 5
    data = data.resample(f"{dt}s").interpolate()
    # fig = px.line(data, x = data.index, y = [
    #     "Total Sky Radiation",
    #     "Dry Bulb Temperature",
    #     ])
    # fig.show()
    del a



    # %%
    daySteps = int(24 * 60 * 60 / dt)
    hStartOffset = 8
    startOffsetSteps = int(hStartOffset * 60 * 60 / dt)
    totalSteps, _ = data.shape



    # %% [markdown]
    #   # BEM Model

    # %% [markdown]
    #   ## Medium Wall

    # %%




    chosenMaterials = getConstructions("Light")

    for material in chosenMaterials:
        print(chosenMaterials[material])

    # plt.figure()
    # for i, material in enumerate([wallMaterial, partitionMaterial, roofMaterial, floorMaterial]):
    #     sns.barplot(x = i, y = material["Deth"], hue = material["Conductivity"])



    # %%
    chosenData = []
    material_types_record = []
    chosenMaterial = []
    floorTempAdjustment = []
    hInterior = []
    hExterior = []
    alphaRoof = []
    windSpeed = []
    wallRoughness = []
    random.seed(randomSeed)
    if N == 1:
        parallel = False
    else:
        parallel = True
    for i in range(N):
        runSteps = int(runDays * daySteps)
        startStep = random.randrange(startOffsetSteps, totalSteps-runSteps-startOffsetSteps, daySteps)
        material_type = random.choice(material_types)
        material_types_record.append(material_type)
        chosenMaterial.append(getConstructions(material_type))
        chosenData.append(data.iloc[startStep : startStep + runSteps])
        floorTempAdjustment.append(random.uniform(-3.5, -5))
        hInterior.append(random.uniform(1, 3))
        alphaRoof.append(random.uniform(0.6, 0.9))
        windSpeed.append(random.uniform(0, 6))
        wallRoughness.append(random.uniform(1.11, 2.17))
        hExterior.append(convectionDOE2(random.uniform(1, 3), windSpeed[i], wallRoughness[i])) #using DOE-2 to calculate this


    inputsMC = [chosenData, chosenMaterial, floorTempAdjustment, hInterior, hExterior, alphaRoof]

    # parallel = False
    realizationOutputs = runMC(inputsMC, parallel = parallel)

    # %%

    if writeResults:
        inputsMCdf = pd.DataFrame(inputsMC[2:], index = ["floorTempAdjustment", "hInterior", "hExterior", "alphaRoof"]).T
        inputsMCdf["material_type"] = material_types_record
        inputsMCdf["windSpeed"] = windSpeed
        inputsMCdf["wallRoughness"] = wallRoughness
        timestr = time.strftime("%Y%m%d-%H%M%S")
        inputsMCdf.to_csv(f"./resultsMC/inputs_{timestr}.csv")

        dfOutputs = pd.DataFrame(realizationOutputs)
        dfOutputs = dfOutputs.stack().apply(pd.Series)
        days = dfOutputs.loc[(slice(None), 'dVent'), :].values.flatten()
        dfOutputs = dfOutputs.unstack(1)
        dfOutputs.to_csv(f"./resultsMC/outputs_{timestr}.csv")
        print(f"time elasped: {time.time() - mainStart}")

    # %%
if __name__ == "__main__":
    main()


