from dask.distributed import Client
import random
from myBuilding import runMyBEM


def runMC(inputs: list, parallel = True):
    if parallel:
        try:
            client.shutdown()
        except:
            pass
        client = Client()
        display(client)

        realizations = client.map(runMyBEM, *inputs, makePlots=False, verbose = False)

        realizationOutputs = client.gather(realizations)

        client.shutdown()
    else:
        for i in range(len(inputs[0])):
            serial_inputs = [input[i] for input in inputs]
            realizationOutputs = runMyBEM(*serial_inputs, makePlots=False, verbose = False)
    
    return realizationOutputs