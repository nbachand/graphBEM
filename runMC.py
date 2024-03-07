from dask.distributed import Client
import random
from myBuilding import runMyBEM


def runMC(weatherData, materials):
    client = Client()
    print(client)

    realizations = client.map(runMyBEM, weatherData, materials, makePlots=False, verbose = False)

    realizationOutputs = client.gather(realizations)

    client.shutdown()
    return realizationOutputs