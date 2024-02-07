import numpy as np
import scipy.linalg as sp_linalg
from model.utils import *

class WallSimulation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["X", "Y", "material_df"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        # Constants
        self.Af = self.X * self.Y #fabric areas
        self.h = WallSides(4, 4) #fabric convection coefficient
        self.alpha = 0.7 #fabric absorptivity
        self.processMaterialDict(self.material_df)
        self.x = np.arange(0, self.th + self.delx, self.delx)

    def processMaterialDict(self, material_df):
        """
        Process pandas df of materials that make up wall
        """
        for index, row in material_df.iterrows():
            if np.isnan(row["Thermal_Resistance"]) == False:
                row["Thickness"] = self.delx # just place one point in the air gap
                row["Conductivity"] = row["Thickness"] / row["Thermal_Resistance"] # convert R value
                row["Density"] = 1 # arbitratily small
                row["Specific_Heat"] = 1.005 # specific heat of air
        material_df["depth"] = material_df["Thickness"].cumsum()
        self.th = np.sum(material_df["Thickness"])
        self.n = 10
        self.delx = self.th / (self.n + 1) # reset delx to evenly divide the thickness

        self.kfs = np.zeros(self.n) #= 2300 #density of fabric
        self.rhofs = np.zeros(self.n) #= 750 #specific heat capacity of fabric
        self.Cfs = np.zeros(self.n) #= 0.8 #thermal conductivity of fabric

        for i in range(self.n):
            depth = (i+1) / (self.n+1) * self.th
            material = material_df[material_df["depth"] >= depth].iloc[0]
            self.kfs[i] = material["Conductivity"]
            self.rhofs[i] = material["Density"]
            self.Cfs[i] = material["Specific_Heat"]


    def initialize(self, delt, TfF, TfB):
        # Scaling factors
        self.lambda_vals = (self.kfs * delt) / (self.rhofs * self.Cfs * self.delx**2)
        self.lambda_bound = WallSides()
        self.lambda_bound.front = self.kfs[0] / (self.h.front * self.delx)
        self.lambda_bound.back = self.kfs[-1] / (self.h.back * self.delx)

        # Wall setup
        self.n = round(self.th / self.delx) - 1
        A_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            A_matrix[(0 + i)%self.n, i] = 1 - 2 * self.lambda_vals[i]
            A_matrix[(1 + i)%self.n, i] = self.lambda_vals[0]
            A_matrix[(-1 + i)%self.n, i] = self.lambda_vals[0]

        A_matrix[0, 0] += self.lambda_vals[0] * self.lambda_bound.front / (1 + self.lambda_bound.front)
        A_matrix[-1, -1] += self.lambda_vals[-1] * self.lambda_bound.back / (1 + self.lambda_bound.back)
        A_matrix[-1, 0] = 0
        A_matrix[0, -1] = 0

        self.A = A_matrix

        self.b = np.zeros(self.n)
        self.T_prof = np.linspace(TfF, TfB, self.n + 2) #create a uniform temperature profile between Tff and Tfb of length n
        self.T = self.T_prof[1:-1] #remove the boundary temperatures from the temperature profile

        self.Erad = WallSides(0, 0) #radiative heat flux at front (area averaged)

    def timeStep(self, TintF, TintB):
        TintRadF = TintF + self.Erad.front / self.h.front
        TintRadB = TintB + self.Erad.back / self.h.back
        self.b[0] = self.lambda_vals[0] * TintRadF / (1 + self.lambda_bound.front)
        self.b[-1] = self.lambda_vals[-1] * TintRadB / (1 + self.lambda_bound.back)
        self.T = np.dot(self.A, self.T) + self.b
        # self.T = np.linalg.solve(self.A, self.b)
        self.T_prof = self.getWallProfile(TintRadF, TintRadB)

        Ef = WallSides()
        # Ef.front = self.Af * (self.T_prof[1] - self.T_prof[0]) / self.delx
        # Ef.back = self.Af * (self.T_prof[-2] - self.T_prof[-1]) / self.delx
        Ef.front = self.Af * (self.T_prof[0] - TintF) * self.h.front
        Ef.back = self.Af * (self.T_prof[-1] - TintB) * self.h.back
        return Ef

    def getWallProfile(self, TintF, TintB):
        T_prof = np.zeros(self.n + 2)
        T_prof[1:-1] = self.T
        T_prof[0] = self.get_Tf(self.T[0], TintF, self.lambda_bound.front)
        T_prof[-1] = self.get_Tf(self.T[-1], TintB, self.lambda_bound.back)
        return T_prof

    def get_Tf(self, T1, Tint, lambda_bound):
        return (lambda_bound * T1 + Tint) / (1 + lambda_bound)