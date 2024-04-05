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

        inputs_futures = []
        for input in inputs:
            inputs_futures.append(client.scatter(input))

        realizations = client.map(runMyBEM, *inputs_futures, makePlots=False, verbose = False)

        realizationOutputs = client.gather(realizations)
        print(realizationOutputs)

        client.shutdown()
    else:
        realizationOutputs = []
        for i in range(len(inputs[0])):
            serial_inputs = [input[i] for input in inputs]
            realizationOutputs.append(runMyBEM(*serial_inputs, makePlots=True, verbose = True))
    
    return realizationOutputs

def main():
    # Call your parallel function
    runMC()

if __name__ == "__main__":
    main()