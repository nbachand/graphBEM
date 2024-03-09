from dask.distributed import Client
import random
from myBuilding import runMyBEM
from IPython.display import display, HTML


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

def main():
    # Call your parallel function
    runMC()

if __name__ == "__main__":
    main()