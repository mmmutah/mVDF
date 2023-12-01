#!/usr/bin/env python
# coding: utf-8


"""
Title: Modified Void Descriptor Function version 2.0
Authors: Dillon S. Watring, John Erickson, Elliott S. Marsden, Ashley D. Spear
License:
  Software License Agreement (BSD License)

  Copyright 2023 D.S. Watring. All rights reserved.
  Copyright 2023 J. Erickson. All rights reserved.
  Copyright 2023 E.S. Marsden. All rights reserved.
  Copyright 2023 A.D. Spear. All rights reserved.
   All rights reserved.

  THE BSD LICENSE

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in
     the documentation and/or other materials provided with the
     distribution.

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
  IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy.spatial.distance import cdist
import math as mt

class poreNetwork:
    """Reads and stores pore network parameters from a .csv file.

    Attributes:
        csv_path: A string indicating the pore network .csv file path.
        csv_data: Data from the .csv file.
        poreNet: A Dictionary of pore parameters.
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.csv_data = self.import_pores()
        self.poreNet = {"pNet": []}
        self.get_CT_params()
        self.get_pore_params()

    def import_pores(self):
        """Imports pore file (please see readme for specifics of pore file)
        Args:
          None

        Returns:
          Data from the .csv pore network file.
        """
        csv_data = genfromtxt(self.csv_path, delimiter=',')
        return csv_data

    def get_CT_params(self):
        """Extracts experiment parameters (e.g. pixel resolution, length of specimen)
        Args:
          None

        Returns:
          None.
        """
        self.p_res = self.csv_data[0, 0]  # Resolution (in um/pixel)
        self.n_slices = int(self.csv_data[0, 1])  # Number of stacks in X-ray CT data/length in um of CT volume (L in VDF equation)
        self.c = self.csv_data[0, 2]  # C value from VDF formulation
        self.gauge_vol = self.csv_data[0, 3]  # Total volume of specimen from CT data
        self.z_ref = np.array([[0.0,0.0,i] for i in range(self.n_slices + 1)])  # Z reference position, which creates an array for the entire length of the specimen
        self.csv_data = np.delete(np.delete(self.csv_data, 1, 0), 0, 0) # Deletes experiment parameters

    def get_pore_params(self):
        """Extracts pore data and organizes parameters into a dictionary.
        Args:
          None

        Returns:
          None
        """
        for i in range(self.csv_data.shape[0]):
            self.poreNet["pNet"].append({
                "pore_ID": 'pore_' + str(i),
                "spher": ((36*mt.pi*(self.csv_data[i,3]**2))**(1/3))/self.csv_data[i,4], #sphericity
                "xc": self.csv_data[i,0]*self.p_res, # X-axis centroid location
                "yc": self.csv_data[i,1]*self.p_res, # Y-axis centroid location
                "zc": self.csv_data[i,2]*self.p_res, # Z-axis (gauge) centroid location
                "vol": self.csv_data[i,3], # Volume
                "r_a": self.csv_data[i,5],  # Semi-major axis a
                "r_b": self.csv_data[i,6],  # Semi-minor axis b
                "r_c": self.csv_data[i,7],  # Semi-minor axis c
                "theta_xy": self.csv_data[i,8],  # Degrees
                "phi_x": self.csv_data[i,10],  # Degrees
                "phi_y": self.csv_data[i,9]  # Degrees
            })

class VDF:
    def __init__(self, net_csv_path, alpha, rho, gamma, zeta):
        """Calculates the VDF along the Z axis.

        The .csv pore network path and VDF fitting parameters are used as input
        for VDF calculation.

        Note: The four fitting parameters (alpha, rho, gamma, zeta) may
            require calibration to your dataset.

        Attributes:
            alpha: A float specifying the axial clustering term fitting parameter.
            rho: A float specifying the free surface term fitting parameter.
            gamma: A float specifying the pore clustering term fitting parameter.
            zeta: A float specifying the stress concentration factor term fitting parameter.
            vdf_dict: A dictionary of pore parameter dictionaries.
            PN: A dictionary of pore parameters dictionaries from the poreNetwork class.
            pores: A list of pore parameter dictionaries.
            z_ref: An array of gauge axis reference locations.
            corner_dist: Distance from the specimen cross-section center to an outside corner.
            v_gauge: Gauge volume of the specimen.
            length: Specimen gauge length.
            kt_min: Minimum pore stress concentration factor in the network.
        """
        self.net_path = net_csv_path

        #user input parameters - recalibrate as needed
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.zeta = zeta

        self.vdf_dict = dict()
        self.PN = poreNetwork(self.net_path)
        self.pores = self.PN.poreNet["pNet"]
        self.z_ref = self.PN.z_ref
        self.corner_dist = self.PN.c
        self.v_gauge = self.PN.gauge_vol
        self.length = self.PN.n_slices
        self.kt_min = 1e9   # Initialize

        #Calculate VDF
        self.read_network()
        self.calc_VDF()

    def read_network(self):
        """Builds dictionary of parameters required for VDF calculation.
        Args:
          None

        Returns:
          None.
        """
        for pn in self.pores:
            self.vdf_dict[pn["pore_ID"]] = dict()
            self.vdf_dict[pn["pore_ID"]]['cen'] = [pn['xc'], pn['yc'], pn['zc']]
            self.vdf_dict[pn["pore_ID"]]['z_loc'] = [0.0, 0.0, pn['zc']]
            self.vdf_dict[pn["pore_ID"]]['vol'] = pn['vol']
            x2 = (pn['xc'] + pn['r_a'] * mt.cos(mt.radians(pn['theta_xy'])) * mt.cos(mt.radians(pn['phi_x']))) ** 2
            y2 = (pn['yc'] + pn['r_a'] * mt.cos(mt.radians(pn['theta_xy'])) * mt.cos(mt.radians(pn['phi_y']))) ** 2
            self.vdf_dict[pn["pore_ID"]]['ri'] = self.corner_dist-mt.sqrt(x2 + y2)
            self.vdf_dict[pn["pore_ID"]]['spher'] = pn['spher']
            self.vdf_dict[pn["pore_ID"]]['Kt'] = self.calc_Kt(pn['r_a'], pn['r_b'], pn['theta_xy'])
        self.kt_min = np.min(np.array([self.vdf_dict[pore["pore_ID"]]['Kt'] for pore in self.pores]))
            
    def calc_Kt(self, r_a, r_b, theta_xy):
        """Builds dictionary of parameters required for VDF calculation.
        Args:
          r_a: pore semi-major axis
          r_b: pore semi-minor axis

        Returns:
          Kt_theta_xy: Interpolated Kt value
        """
        Kt_Beta_0 = 1.0 + 2 * (r_b / float(r_a))
        Kt_Beta_90 = 1.0 + 2 * (r_a / float(r_b))
        Kt_theta_xy = (Kt_Beta_90 - Kt_Beta_0) * ((90 - theta_xy) / 90.0) + Kt_Beta_0
        return Kt_theta_xy

    def calc_si(self):
        """Calculates the axial clustering term.
        Args:
          None

        Returns:
          None
        """
        self.z_locs = np.array([self.vdf_dict[pore["pore_ID"]]['z_loc'] for pore in self.pores])
        self.si = abs(cdist(self.z_ref, self.z_locs, 'euclidean'))
        self.si_term = self.si / (self.alpha * self.length)
        self.si_term *= -1.0

    def calc_ai(self):
        """Calculates the pore clustering term.
        Args:
          None

        Returns:
          None
        """
        self.cents = np.array([self.vdf_dict[pore["pore_ID"]]['cen'] for pore in self.pores])
        dists = cdist(self.cents, self.cents, 'euclidean')
        # Linear weighting######
        dists.sort(axis=1)
        dists = np.delete(dists, 0, 1)
        dists_copy = np.copy(dists)
        max_all_d = np.max(np.sum(dists_copy, axis=0)) # This is a constant
        # Creates a linear array from 1 to 0
        lin_array = np.linspace(1.0, 0.0, num=dists.shape[1])
        # Calculates the weighted distance array
        dists *= lin_array
        sum_of_dists = np.sum(dists, axis=1)

        for i, pn in enumerate(self.pores):
            self.vdf_dict[pn["pore_ID"]]['we_d'] = sum_of_dists[i]
        self.ai_term = np.array([self.vdf_dict[pore["pore_ID"]]['we_d'] / (max_all_d * self.gamma) for pore in self.pores])

    def calc_ci(self):
        """Calculates the distance to free surface term.
        Args:
          None

        Returns:
          None
        """
        self.ci_term = np.array([self.vdf_dict[pore["pore_ID"]]['ri'] / (self.rho * self.corner_dist) for pore in self.pores])

    def calc_ki(self):
        """Calculates the shape and orientation stress concentration term.
        Args:
          None

        Returns:
          None
        """
        self.ki_term = np.array([self.vdf_dict[pore["pore_ID"]]['spher'] * ((self.kt_min/(self.vdf_dict[pore["pore_ID"]]['Kt']*self.zeta))) for pore in
                       self.pores])
        
    def calc_VDF(self):
        """Calculates the VDF and determines the max VDF index location.
        Args:
          None

        Returns:
          None
        """
        self.calc_si() # Calculates the axial clustering term
        self.calc_ci() # Calculates the free surface term
        self.calc_ai() # Calculates the 3D clustering term
        self.calc_ki() # Calculates the stress concentration term
        self.pore_vols = np.array([self.vdf_dict[pore["pore_ID"]]['vol'] / self.v_gauge for pore in self.pores]) # All pore volumes normalized by the gauge volume
        self.exp_func = np.copy(self.si_term) # Exponential weighting function
        self.exp_func -= self.ci_term
        self.exp_func -= self.ai_term
        self.exp_func -= self.ki_term
        self.exp_func = np.exp(self.exp_func)
        self.exp_func *= self.pore_vols
        self.vdf_values = np.sum(self.exp_func, axis=1) # Final VDF values for all Z_ref locations
        self.max_location = np.argmax(self.vdf_values) # Index of the maximum VDF value

    def plot_VDF(self, norm=False):
        """Plots the VDF as a function of Z axis position.
        Args:
          norm: A boolean that controls normalization of the VDF distribution.

        Returns:
          None
        """
        zs = np.array([i[2] for i in self.z_ref])
        net_name = self.net_path.split('/')
        if norm:
          plt.plot(zs, self.vdf_values/np.max(self.vdf_values))
        else:
          plt.plot(zs, self.vdf_values)
        plt.title(f'{net_name[-1]}')
        plt.xlabel('z-ref locations')
        plt.ylabel('VDF values')
        plt.show()

#********** User inputs **********
# Default parameters are corrected from Watring, et al., Acta Materialia (2022)
# Note that your problem set might require recalibration of these parameters to optimize performance
alpha = 0.342
rho = 1.0
gamma = 0.1
zeta = 0.1
#*********************************
if __name__ == "__main__":
    network_path = 'example.csv' # Input your pore file path here
    network_VDF = VDF(network_path, alpha, rho, gamma, zeta)
    print(f'Max VDF value is {np.max(network_VDF.vdf_values)} at slice {np.argmax(network_VDF.vdf_values)}')
    network_VDF.plot_VDF()





