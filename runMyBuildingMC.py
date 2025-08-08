# %%

import numpy as np
from matplotlib import pyplot as plt
from runMC import runMC
from model.utils import *
from model.WallSimulation import convectionDOE2
import random
import os
import time
from energyPlus.weather.weather import *

plt.close('all')

def setConstructionType(materials, type):
    R_values = {
        "Heavy": {
            "wall": 20,
            "roof": 38,
            "floor": 28,
        },
        "Medium": {
            "wall": 13,
            "roof": 26,
            "floor": 20,
        },
        "Light": {
            "wall": 8,
            "roof": 20,
            "floor": 13,
        }
    }
    materials["wall"] = adjustConstructionRValue(R_values[type]["wall"], materials["wall"],  "I02 50mm insulation board", adjustConductivity=False)
    materials["roof"] = adjustConstructionRValue(R_values[type]["roof"], materials["roof"],  "I05 154mm batt insulation", adjustConductivity=False)
    materials["floor"] = adjustConstructionRValue(R_values[type]["floor"], materials["floor"],  "I05 154mm batt insulation", adjustConductivity=False)
    for material in materials:
        materials[material]["depth"] = materials[material]["Thickness"].cumsum()

    return materials

def adjustConstructionRValue(desired_r_value, construction_df, material_name, adjustConductivity = False):
    construction_df_copy = construction_df.copy()
    
    for material in construction_df_copy.index.values:
        if np.isnan(construction_df_copy.loc[material, 'Thermal_Resistance']):
            construction_df_copy.loc[material, 'Thermal_Resistance'] = construction_df_copy.loc[material, 'Thickness'] / construction_df_copy.loc[material, 'Conductivity']

    if "Soil" in construction_df_copy.index.values:
        desired_r_value += construction_df_copy.loc["Soil", 'Thermal_Resistance'] * 5.678
        # display(f"Desired R-value adjusted for soil: {desired_r_value}")
    

    # Calculate the initial R-value of the construction
    initial_resistance = construction_df_copy['Thermal_Resistance'].sum()
    
    # Calculate the initial contribution of the material to the initial R-value
    initial_material_contribution = construction_df_copy.loc[material_name, 'Thermal_Resistance']

    other_material_contributions = initial_resistance - initial_material_contribution
    
    # Calculate the desired contribution of the material to achieve the desired total R-value
    desired_material_contribution = desired_r_value / 5.678 - other_material_contributions
    
    if adjustConductivity:
        new_conductivity = construction_df_copy.loc[material_name, 'Thickness'] / desired_material_contribution
        construction_df_copy.loc[material_name, 'Conductivity'] = new_conductivity 
    else:
        # Calculate the adjustment factor for the thickness of the specified material
        new_thickness = desired_material_contribution * construction_df_copy.loc[material_name, 'Conductivity']
    
        # Adjust the thickness of the specified material
        construction_df_copy.loc[material_name, 'Thickness'] = new_thickness

    construction_df_copy.loc[material_name, 'Thermal_Resistance'] =construction_df_copy.loc[material_name, 'Thickness'] / construction_df_copy.loc[material_name, 'Conductivity']

    try:
        assert construction_df_copy.loc[material_name, 'Thermal_Resistance'] >= desired_material_contribution - .001 and construction_df_copy.loc[material_name, 'Thermal_Resistance'] < desired_material_contribution + .001
        assert construction_df_copy['Thermal_Resistance'].sum() * 5.678 >= desired_r_value - .001 and construction_df_copy['Thermal_Resistance'].sum() * 5.678 < desired_r_value + .001
    except AssertionError:
        print(f"Desired R-value not achieved. Error in calculation")
    try:
        assert construction_df_copy.loc[material_name, 'Thickness'] >= 0
        assert construction_df_copy.loc[material_name, 'Conductivity'] >= 0
    except AssertionError:
        print(f"Negative thickness or conductivity found")
    
    # Return the modified construction dataframe with adjusted thickness
    return construction_df_copy

def getConstructions(type, soil_depth = 0.5, soil_conductivity = 1.5, soil_density = 2800, soil_specific_heat = 850, constructionFile = "energyPlus/ASHRAE_2005_HOF_Constructions.csv", materialFile = "energyPlus/ASHRAE_2005_HOF_Materials.csv"):
    floor = cleanMaterial(f"{type} Floor", reverse=False, constructrionFile=constructionFile, materialFile=materialFile)
    floor.loc["Soil"] = ["Material", np.nan, soil_depth, soil_conductivity, soil_density, soil_specific_heat, np.nan]
    return {
        "wall": cleanMaterial(f"{type} Exterior Wall", constructrionFile=constructionFile, materialFile=materialFile),
        "partition": cleanMaterial(f"{type} Partitions", constructrionFile=constructionFile, materialFile=materialFile),
        "roof": cleanMaterial(f"{type} Roof/Ceiling", constructrionFile=constructionFile, materialFile=materialFile),
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

def cleanMaterial(materialName, reverse = True, constructrionFile = "energyPlus/ASHRAE_2005_HOF_Constructions.csv", materialFile = "energyPlus/ASHRAE_2005_HOF_Materials.csv"):
    constructions  = pd.read_csv(constructrionFile, index_col="Name")
    materials = pd.read_csv(materialFile, index_col="Name")
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

def main(N = 300, runDays = 7, resultsKey = "timestr", randomSeed = 666, material_types = ["Light", "Medium", "Heavy"]):

    mainStart = time.time()

    # %% [markdown]
    #   # Specify Weather Data

    # %%

    # Generate sinusoidal data
    times, Touts, rad = generate_sinusoidal_data(delt=5, length=96)




    # %%
    data, climate_zones = getWeatherData()
    dt = 30



    # %% [markdown]
    #   # BEM Model

    # %% [markdown]
    #   ## Medium Wall

    # %%
    materials = getConstructions("My", constructionFile = "energyPlus/My_Constructions.csv")
    # materials = getConstructions("Light", constructionFile = "energyPlus/ASHRAE_2005_HOF_Constructions.csv")

    # plt.figure()
    # for i, material in enumerate([wallMaterial, partitionMaterial, roofMaterial, floorMaterial]):
    #     sns.barplot(x = i, y = material["Deth"], hue = material["Conductivity"])



    # %%
    weatherPropertiesRecord = []
    chosenData = []
    materialTypesRecord = []
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
        material_type = random.choice(material_types)
        materialTypesRecord.append(material_type)
        chosenMaterial.append(setConstructionType(materials, material_type))
        floorTempAdjustment.append(random.uniform(-3.5, -5))
        hInterior.append(random.uniform(1, 3))
        alphaRoof.append(random.uniform(0.6, 0.9))
        windSpeed.append(random.uniform(0, 6))
        wallRoughness.append(random.uniform(1.11, 2.17))
        hExterior.append(convectionDOE2(random.uniform(1, 3), windSpeed[i], wallRoughness[i])) #using DOE-2 to calculate this

        weatherProperties, dataSampled = sampleVentWeather(data, climate_zones, runDays, dt=dt, plot=False)
        weatherPropertiesRecord.append(weatherProperties)
        dataSampled = dataSampled.infer_objects(copy=False)
        dataSampled = dataSampled.resample(f"{dt}s").interpolate()
        chosenData.append(dataSampled)

    print("Generated BEM inputs")
    inputsMC = [chosenData, chosenMaterial, floorTempAdjustment, hInterior, hExterior, alphaRoof]

    # parallel = False
    realizationOutputs = runMC(inputsMC, parallel = parallel)

    # %%

    if resultsKey != None:
        if resultsKey == "timestr":
            key = time.strftime("%Y%m%d-%H%M%S")
        else:
            key = resultsKey
        inputsMCdf = pd.DataFrame(inputsMC[2:], index = ["floorTempAdjustment", "hInterior", "hExterior", "alphaRoof"]).T
        inputsMCdf["weatherProperties"] = weatherPropertiesRecord
        inputsMCdf["material_type"] = materialTypesRecord
        inputsMCdf["windSpeed"] = windSpeed
        inputsMCdf["wallRoughness"] = wallRoughness
        inputsMCdf.to_csv(f"./resultsMC/inputs_{key}.csv")

        dfOutputs = pd.DataFrame(realizationOutputs)
        dfOutputs = dfOutputs.stack().apply(pd.Series)
        days = dfOutputs.loc[(slice(None), 'dVent'), :].values.flatten()
        dfOutputs = dfOutputs.unstack(1)
        dfOutputs.to_csv(f"./resultsMC/outputs_{key}.csv")
        print(f"time elasped: {time.time() - mainStart}")

    # %%
if __name__ == "__main__":
    main()


