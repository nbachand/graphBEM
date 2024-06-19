from epw import epw
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import plotly.express as px
import random
import seaborn as sns

def process_epw_file(file_path, verbose=False):
    # Initialize the EPW object
    a = epw()
    
    # Read the EPW file
    a.read(file_path)
    
    # Set the dataframe index to a datetime format
    a.dataframe.index = pd.to_datetime(a.dataframe[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    
    if verbose:
        print(f"Data for months: {set(a.dataframe['Month'])}, years: {set(a.dataframe['Year'])}")
        # Display the keys of the dataframe
        keys = a.dataframe.keys()
        print("DataFrame Keys:", keys)
        
        # Display the unique years in the dataframe
        unique_years = set(a.dataframe["Year"])
        print("Unique Years:", unique_years)
    
        # Extract ground temperatures from headers
        ground_temps = a.headers["GROUND TEMPERATURES"]
        print("Number of Ground Temperature Entries:", len(ground_temps))
    
    # Calculate unique months
    uniqueMonths = a.dataframe.index[a.dataframe.index.day==15]
    uniqueMonths = uniqueMonths.year.values + uniqueMonths.month.values/100
    uniqueMonths = np.unique(uniqueMonths)
    num_unique_months = len(uniqueMonths)
    if num_unique_months != 12:
        raise ValueError(f"Number of unique months is not 12. Found {num_unique_months} unique months.")
    # Return the processed information
    return a.dataframe
    
def getWeatherData(climateZoneKey = "CA_climate_zones.csv", verbose=False):
    climate_zones = {
        1: {
            "City": "Arcata",
            "Latitude": 41.0,
            "Longitude": 124.1,
            "Elevation": 203,
            "WeatherFile": "CA_ARCATA-AP_725945S_CZ2022"
        },
        2: {
            "City": "Santa Rosa",
            "Latitude": 38.5,
            "Longitude": 122.8,
            "Elevation": 125,
            "WeatherFile": "CA_SANTA-ROSA-AP_724957S_CZ2022"
        },
        3: {
            "City": "Oakland",
            "Latitude": 37.7,
            "Longitude": 122.2,
            "Elevation": 6,
            "WeatherFile": "CA_OAKLAND-METRO-AP_724930S_CZ2022"
        },
        4: {
            "City": "San Jose-Reid",
            "Latitude": 37.3,
            "Longitude": 121.8,
            "Elevation": 135,
            "WeatherFile": "CA_SAN-JOSE-REID-HILLV_724946S_CZ2022"
        },
        5: {
            "City": "Santa Maria",
            "Latitude": 34.9,
            "Longitude": 120.4,
            "Elevation": 253,
            "WeatherFile": "CA_SANTA-MARIA-PUBLIC-AP_723940S_CZ2022"
        },
        6: {
            "City": "Torrance",
            "Latitude": 33.8,
            "Longitude": 118.3,
            "Elevation": 88,
            "WeatherFile": "CA_TORRANCE-MUNI-AP_722955S_CZ2022"
        },
        7: {
            "City": "San Diego-Lindbergh",
            "Latitude": 32.7,
            "Longitude": 117.2,
            "Elevation": 13,
            "WeatherFile": "CA_SAN-DIEGO-LINDBERGH-FLD_722900S_CZ2022"
        },
        8: {
            "City": "Fullerton",
            "Latitude": 33.9,
            "Longitude": 118.0,
            "Elevation": 95,
            "WeatherFile": "CA_FULLERTON-MUNI-AP_722976S_CZ2022"
        },
        9: {
            "City": "Burbank-Glendale",
            "Latitude": 34.2,
            "Longitude": 118.3,
            "Elevation": 741,
            "WeatherFile": "CA_BURBANK-GLNDLE-PASAD-AP_722880S_CZ2022"
        },
        10: {
            "City": "Riverside",
            "Latitude": 33.9,
            "Longitude": 117.4,
            "Elevation": 840,
            "WeatherFile": "CA_RIVERSIDE-MUNI_722869S_CZ2022"
        },
        11: {
            "City": "Red Bluff",
            "Latitude": 40.1,
            "Longitude": 122.2,
            "Elevation": 348,
            "WeatherFile": "CA_RED-BLUFF-MUNI-AP_725910S_CZ2022"
        },
        12: {
            "City": "Sacramento",
            "Latitude": 38.5,
            "Longitude": 121.5,
            "Elevation": 16,
            "WeatherFile": "CA_SACRAMENTO-EXECUTIVE-AP_724830S_CZ2022"
        },
        13: {
            "City": "Fresno",
            "Latitude": 36.8,
            "Longitude": 119.7,
            "Elevation": 335,
            "WeatherFile": "CA_FRESNO-YOSEMITE-IAP_723890S_CZ2022"
        },
        14: {
            "City": "Palmdale",
            "Latitude": 34.6,
            "Longitude": 118.0,
            "Elevation": 2523,
            "WeatherFile": "CA_PALMDALE-AP_723820S_CZ2022"
        },
        15: {
            "City": "Palm Springs-Intl",
            "Latitude": 33.8,
            "Longitude": 116.5,
            "Elevation": 475,
            "WeatherFile": "CA_PALM-SPRINGS-IAP_722868S_CZ2022"
        },
        16: {
            "City": "Blue Canyon",
            "Latitude": 39.2,
            "Longitude": 120.7,
            "Elevation": 5279,
            "WeatherFile": "CA_BLUE-CANYON-AP_725845S_CZ2022"
        }
    }

    data = pd.DataFrame()
    for zone, info in climate_zones.items():
        epw_file = f"./energyPlus/weather/CAClimateZones/{info['WeatherFile']}/{info['WeatherFile']}.epw"
        zoneData = process_epw_file(epw_file, verbose=verbose)
        zoneData["City"] = info["City"]
        zoneData["Latitude"] = info["Latitude"]
        zoneData["Longitude"] = info["Longitude"]
        zoneData["Elevation"] = info["Elevation"]
        zoneData["ClimateZone"] = zone
        data = pd.concat([data, zoneData], axis="index")
    
    return data, climate_zones

def sampleVentWeather(data, climate_zones, runDays, dt, plot=False):
    # Constants
    dt = 3600  # Data time step in seconds
    coolingThreshold = 24
    coolingDegBase = 21
    daySteps = 24 * 60 * 60 // dt  # Number of time steps in a day
    hStartOffset = 8  # Start offset in hours
    startOffsetSteps = hStartOffset * 60 * 60 // dt  # Start offset in steps
    weatherDays = runDays + int(np.ceil(hStartOffset / 24))  # Total days including offset
    runSteps = runDays * daySteps  # Total run steps
    weatherSteps = weatherDays * daySteps  # Total weather steps

    foundVentWeatherData = False
    while foundVentWeatherData == False:
        # Randomly select a climate zone and month
        chosenZone = random.choice(list(climate_zones.keys()))
        chosenMonth = random.randint(1, 12)

        # Filter data for the chosen zone and month
        dataSampled = data[(data["ClimateZone"] == chosenZone) & (data.index.month == chosenMonth)]

        # Determine the most common year in the data
        year = sp.stats.mode(dataSampled.index.year.values).mode
        dataSampled = dataSampled[dataSampled.index.year == year]

        # Select a random starting step
        totalSteps, _ = dataSampled.shape
        startStep = random.randrange(0, totalSteps - weatherSteps, daySteps)

        # Select the data for the weather period
        dataSampled = dataSampled.iloc[startStep : startStep + weatherSteps]

        # Resample the data to daily highs, lows, and average wind speed
        daily_highs = dataSampled.resample('D')['Dry Bulb Temperature'].max()[0:-1]
        daily_lows = dataSampled.resample('D')['Dry Bulb Temperature'].min()
        daily_wind = dataSampled.resample('D')['Wind Speed'].mean()

        # Initialize lists to store daily values
        cooling_degree = []
        vent_degree = []
        vent_wind = []

        # Calculate cooling degree days and ventilation degree days
        for i in range(runDays):
            avg_temp = (daily_highs.iloc[i] + daily_lows.iloc[i]) / 2
            cooling_degree_day = max(0, avg_temp - coolingDegBase)
            cooling_degree.append(cooling_degree_day)
        
            if daily_lows.iloc[i + 1] < coolingThreshold:
                vent_degree.append(cooling_degree_day)
                vent_wind.append(daily_wind.iloc[i])
            else:
                vent_degree.append(0)

        # Calculate total cooling degree days and ventilation degree days
        cooling_degree_days = np.sum(cooling_degree)
        vent_degree_days = np.sum(vent_degree)
        if vent_degree_days >= runDays:
            foundVentWeatherData = True

    # Append the calculated values to the weatherProperties dictionary
    weatherProperties = {
        "zone": chosenZone,
        "month": chosenMonth,
        "cooling_degree_days": cooling_degree_days,
        "ventilation_degree_days": vent_degree_days,
        "ventilation_wind": np.mean(vent_wind)
    }

    # Select the data for the run period
    dataSampled = dataSampled.iloc[startOffsetSteps : runSteps + startOffsetSteps]

    if plot:
        plt.figure(figsize=(12, 6))
    
        # Plot the dry bulb temperature over time
        plt.plot(dataSampled.index, dataSampled["Dry Bulb Temperature"], label='Dry Bulb Temperature')
    
        # Scatter plot for daily highs and lows
        plt.scatter(daily_highs.index, daily_highs, color='red', label='Daily Highs')
        plt.scatter(daily_lows.index, daily_lows, color='blue', label='Daily Lows')
    
        # Scatter plot for cooling degree and ventilation degree
        for i in range(runDays):
            if cooling_degree[i] > 0:
                plt.scatter(daily_highs.index[i], daily_lows.iloc[i] + cooling_degree[i], color='orange', label='Cooling Degree' if i == 0 else "")
            if vent_degree[i] > 0:
                plt.scatter(daily_highs.index[i], daily_lows.iloc[i] + vent_degree[i], color='green', label='Ventilation Degree' if i == 0 else "")
                plt.scatter(daily_highs.index[i], vent_wind[i] * 10, color='black', label='Ventilation Wind' if i == 0 else "")
    
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C) / Wind Speed (m/s) x 10')
        plt.title('Weather Data with Cooling and Ventilation Degrees')
        plt.legend()
        plt.grid(True)

    return weatherProperties, dataSampled
