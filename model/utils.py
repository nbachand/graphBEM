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

def getSide(front, back, node, throwException = True):
    if front == node:
        return front
    elif back == node:
        return back
    else:
        if throwException:
            raise Exception("Node not found in edge")
        else:
            return ''
        

class WallSides:
    """
    Class to keep track of both wall surfaces and providing some useful methods
    """
    def __init__(self, front = None, back = None):
        self.front = front
        self.back = back
        self.updateFront = False
        self.updateBack = False

    def setUpdateFront(self):
        self.updateFront = True
        self.updateBack = False

    def setUpdateBack(self):
        self.updateFront = False
        self.updateBack = True

    def setUpdateBoth(self):
        self.updateFront = True
        self.updateBack = True

    def update(self, val, reset = True):
        if self.updateFront is True:
            self.front = val
        elif self.updateBack is True:
            self.back = val
        self.updateFront = False
        self.updateBack = False

    def checkSides(self, side, throwException = True):
        return getSide(self.front, self.back, side, throwException = throwException)
    
    def getSideIndex(self, side, reverse = False):
        isFront = (side == self.front)
        isBack = (side == self.back)
        frontIndex = 0 - reverse
        backIndex = -1 + reverse
        if isFront and isBack:
            return 999 # code for both sides
        if isFront:
            return frontIndex # if reversed, front is in other room so return -1
        if isBack: 
            return backIndex # if reversed, back is in other room so return 0
        raise Exception("Node not found in edge")
