import numpy as np
import scipy.linalg as sp_linalg
from model.utils import *

def convectionDOE2(h_nat, V_10, R_f):
    """
    Calculate the convection coefficient using the DOE-2 method
    """
    alpha = np.mean([2.38, 2.86])
    beta = np.mean([0.617, 0.89])
    return (1 - R_f) * h_nat + R_f * (h_nat**2 + (alpha * V_10**beta)**2)**0.5
    
def processMaterials(material_df, n, dt = None, verbose = True):
    """
    Process pandas df of materials that make up wall
    """

    th = np.sum(material_df["Thickness"])
    delx = th / (n + 1) # set delx to evenly divide the thickness
    material_df["n"] = None
    material_df.reset_index(drop = False, inplace = True)
    for i, (index, row) in enumerate(material_df.iterrows()):
        if row["key"] == "Material:AirGap":
            n += 1 # add node in air gap
            material_df.loc[index, "Thickness"] = delx # just place one point in the air gap
            material_df.loc[index, "Conductivity"] = material_df.loc[index, "Thickness"] / material_df.loc[index, "Thermal_Resistance"] # convert R value
            material_df.loc[index, "Density"] = 1.293 # * 50 # large enough for reasonable time step (50X)
            material_df.loc[index, "Specific_Heat"] = 1005 # specific heat of air
            material_df.loc[index, "n"] = 1
        else: # make sure material thicknesses align with spacial discretization and adjust properties accordingly
            nMat = max([1, round(material_df.loc[index, "Thickness"] / delx)]) # make sure n is at least 1
            material_df.loc[index, "n"] = nMat
            new_thickness = nMat * delx
            scaling_factor = material_df.loc[index, "Thickness"] / new_thickness
            material_df.loc[index, "Conductivity"] = material_df.loc[index, "Conductivity"] * scaling_factor
            material_df.loc[index, "Specific_Heat"] = material_df.loc[index, "Specific_Heat"] * scaling_factor
            material_df.loc[index, "Thickness"] = new_thickness
            material_df.loc[index, "Thermal_Resistance"] =  material_df.loc[index, "Thickness"] / material_df.loc[index, "Conductivity"] # convert to R value for reference
        if dt is not None:
            densityMargin = 1.1
            densityMin = densityMargin * dt * material_df.loc[index, "Conductivity"] / (delx**2 * material_df.loc[index, "Specific_Heat"])
            if densityMin > material_df.loc[index, "Density"]:
                if verbose:
                    print(f"WARNING: Material {material_df['index'][index]} has density of {material_df.loc[index, 'Density']} but should be at least {densityMin} for time step {dt} ({int(densityMin/material_df.loc[index, 'Density'])}X)")
                material_df.loc[index, "Density"] = densityMin
    material_df.set_index("index", inplace = True)
    material_df["depth"] = material_df["Thickness"].cumsum()

    return material_df


class WallSimulation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["X", "Y", "material_df", "h", "absorptivity", "n", "delt"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        # Constants
        self.Af = self.X * self.Y #fabric areas
        self.processMaterialDict(self.material_df)
        self.x = np.linspace(0, self.th, self.n + 2)

    def processMaterialDict(self, material_df):
        material_df = processMaterials(material_df, self.n, dt = self.delt)
        # self.n  = int(material_df["n"].sum())
        self.th = np.sum(material_df["Thickness"])
        self.delx = self.th / (self.n + 1) # set delx to evenly divide the thickness
        self.kfs = np.zeros(self.n) #= 2300 #density of fabric
        self.rhofs = np.zeros(self.n) #= 750 #specific heat capacity of fabric
        self.Cfs = np.zeros(self.n) #= 0.8 #thermal conductivity of fabric

        depth = self.delx / 2
        for i in range(self.n):
            depth += self.delx
            material = material_df[material_df["depth"] >= depth].iloc[0]
            self.kfs[i] = material["Conductivity"]
            self.rhofs[i] = material["Density"]
            self.Cfs[i] = material["Specific_Heat"]

        self.material_df = material_df

    def initialize(self, delt, TfF, TfB, verbose = False):
        # Scaling factors
        self.lambda_vals = (delt / self.delx**2) * self.kfs / (self.rhofs * self.Cfs)
        if verbose:
            i_max = np.argmax(self.lambda_vals)
            # print(f"maximum time step: {delt/self.lambda_vals[i_max]} at node {i_max}")
        # create error to catch timestep that is too large
        try:
            assert np.min(delt/self.lambda_vals) > delt
        except:
            raise ValueError("Time step too large for stability")
        self.lambda_bound = WallSides()
        self.lambda_bound.front = self.kfs[0] / (self.h.front * self.delx)
        self.lambda_bound.back = self.kfs[-1] / (self.h.back * self.delx)

        # Wall setup
        self.n = round(self.th / self.delx) - 1
        A_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            A_matrix[i, i] = 1 - 2 * self.lambda_vals[i]
            if i < self.n - 1:
                A_matrix[i, i + 1] = self.lambda_vals[i]
            if i > 0:
                A_matrix[i, i - 1] = self.lambda_vals[i]

        A_matrix[0, 0] += self.lambda_vals[0] * self.lambda_bound.front / (1 + self.lambda_bound.front)
        A_matrix[-1, -1] += self.lambda_vals[-1] * self.lambda_bound.back / (1 + self.lambda_bound.back)

        self.A = A_matrix

        # ###########################
        # # Plot the color plot
        # import plotly.graph_objs as go
        # import plotly.express as px
        # # Create a heatmap figure
        # fig = go.Figure(data=go.Heatmap(z=self.A, colorscale='Viridis'))

        # # Update layout
        # fig.update_layout(
        #     title='Heatmap of Matrix',
        #     xaxis_title='X-axis',
        #     yaxis_title='Y-axis'
        # )

        # # Show the figure
        # fig.show()
        # ###########################

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
        Tf = (lambda_bound * T1 + Tint) / (1 + lambda_bound)
        if np.isnan(Tf) == True:
            raise ValueError('Tf is nan')
        return Tf