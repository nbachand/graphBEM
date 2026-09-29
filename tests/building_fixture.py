"""Construct the production example with deterministic synthetic weather."""
from unittest.mock import patch
import pandas as pd
from myBuilding import runMyBEM
from runMyBuildingMC import getConstructions
from model.BuildingSimulation import BuildingSimulation


def example(dt=30, steps=4):
    weather = pd.DataFrame(index=pd.date_range('2008-08-01', periods=steps+1,
                                             freq=f'{dt}s', tz='Etc/GMT+8'))
    for col, value in [('temp_air', 20.), ('wind_speed', 1.), ('Latitude', 34.2),
                       ('Horizontal Shortwave Radiation', 0.), ('Vertical Shortwave Radiation', 0.),
                       ('Horizontal Sky Longwave Radiation', 350.), ('Vertical Sky Longwave Radiation', 175.),
                       ('Horizontal Surfaces Longwave Radiation', 0.), ('Vertical Surfaces Longwave Radiation', 230.)]:
        weather[col] = value
    captured = []
    class Initialized(Exception):
        pass
    def capture(sim):
        captured.append(sim)
        raise Initialized
    materials = getConstructions('My', constructionFile='energyPlus/My_Constructions.csv')
    try:
        with patch.object(BuildingSimulation, 'run', capture):
            runMyBEM(weather, materials, -4.25, 2, 2, 1.64, .75)
    except Initialized:
        pass
    return captured[0]
