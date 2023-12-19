import numpy as np
import pandas as pd

def getEquivalentTimeSeries(x, tSeries):
    if isinstance(x, float) or isinstance(x, int):
        x = np.ones_like(tSeries) * x
    elif x is None:
        x =  np.zeros_like(tSeries)
    elif len(x) != len(tSeries):
        raise Exception("x and tSeries must be the same length")
    return x

# Function to build the system of equations
def graphToSysEqnKCL(graph):
    A = pd.DataFrame(0.0, graph.nodes, columns=graph.nodes)

    # Build equations based on KCL for each node (except the reference node)
    for n, d in graph.nodes(data=True):
        # Sum of currents entering the node equals the sum of currents leaving the node (KCL)
        A[n][n] += 1
        for e in graph[n]:
            A[e][n] = -d["boundaryResistance"] / graph[n][e]['radianceResistance']
            A[n][n] += d["boundaryResistance"] / graph[n][e]['radianceResistance']

    return A

class WallFlux:
    def __init__(self, front = None, back = None):
        self.front = front
        self.back = back