# %%
%load_ext autoreload
%autoreload 2
%matplotlib widget

import numpy as np
from matplotlib import pyplot as plt
from myBuilding import runMyBEM
from model.utils import *
import matplotlib.colors as mcolors
from epw import epw
import plotly.express as px
import dask
import random

plt.close('all')

# %% [markdown]
# # Specify Weather Data

# %%
# Function to generate sinusoidal data for temperature and radiation
def generate_sinusoidal_data(delt = 15, amplitude_temp=10, amplitude_radiation=250, length=24):
    length *= 3600  # Converting period from hours to seconds
    period = 24 * 3600
    time = np.arange(0, length, delt)  # Representing a day (24 hours) with a sinusoidal pattern
    temperature = amplitude_temp * np.sin(2 * np.pi * time / period) + 300  # Centered 
    radiation = amplitude_radiation * np.sin(2 * np.pi * time / period) + 250
    return time, temperature, radiation

# Generate sinusoidal data
times, Touts, rad = generate_sinusoidal_data(delt=5, length=96)


# %%
a = epw()
a.read("./energyPlus/weather/USA_CA_Sacramento.724835_TMY2.epw")
a.dataframe.index = pd.to_datetime(a.dataframe[['Year', 'Month', 'Day', 'Hour', 'Minute']])
display(a.dataframe.keys())
display(set(a.dataframe["Year"]))

# %%
data = a.dataframe[(a.dataframe["Month"]==8) & (a.dataframe["Day"] >= 1)]
data["Total Sky Radiation"] = data["Horizontal Infrared Radiation Intensity"] + data["Global Horizontal Radiation"]
dt = 5
data = data.resample(f"{dt}s").interpolate()
fig = px.line(data, x = data.index, y = [
    "Total Sky Radiation",
    "Dry Bulb Temperature",
    ])
fig.show()

# %%
daySteps = int(24 * 60 * 60 / dt)
totalSteps, _ = data.shape

# %% [markdown]
# # BEM Model

# %% [markdown]
# ## Medium Wall

# %%
constructions  = pd.read_csv("energyPlus/ASHRAE_2005_HOF_Constructions.csv", index_col="Name")
materials = pd.read_csv("energyPlus/ASHRAE_2005_HOF_Materials.csv", index_col="Name")


def cleanMaterial(materialName, reverse = True):
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

def getConstructions(type, soil_depth = 0.5, soil_conductivity = 1.5, soil_density = 2800, soil_specific_heat = 850):
    floor = cleanMaterial(f"{type} Floor", reverse=False)
    floor.loc["Soil"] = ["Material", np.nan, soil_depth, soil_conductivity, soil_density, soil_specific_heat, np.nan]
    return {
        "wall": cleanMaterial(f"{type} Exterior Wall"),
        "partition": cleanMaterial(f"{type} Partitions"),
        "roof": cleanMaterial(f"{type} Roof/Ceiling"),
        "floor": floor
    }

chosenMaterials = getConstructions("Light")

for material in chosenMaterials:
    display(chosenMaterials[material])

# plt.figure()
# for i, material in enumerate([wallMaterial, partitionMaterial, roofMaterial, floorMaterial]):
#     sns.barplot(x = i, y = material["Deth"], hue = material["Conductivity"])

# %%
try:
    client.shutdown()
except:
    pass
from dask.distributed import Client
client = Client()
client

# %%
N = 8
material_types = ["Light", "Medium", "Heavy"]
runDays = 1 # day
runSteps = int(runDays * daySteps)
realizations = []
chosenData = []
chosenMaterial = []
for i in range(N):
    startStep = random.randrange(0, totalSteps-runSteps, daySteps)
    material_type = random.choice(material_types)
    chosenMaterial.append(getConstructions(material_type))
    chosenData.append(data.iloc[startStep : startStep + runSteps])

realizations = client.map(runMyBEM, chosenData, chosenMaterial, makePlots=False, verbose = False)

realizationOutputs = client.gather(realizations)


# %%
dfOutputs = pd.DataFrame(realizationOutputs)



# %%
fig = px.histogram(dfOutputs,
    marginal="box", # or violin, rug
    barmode = "group"
    )
fig.show()


