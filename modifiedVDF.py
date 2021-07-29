#!/usr/bin/env python
# coding: utf-8


"""
Title: Modified Void Descriptor Function
Authors: Dillon S. Watring, John Erickson, Ashley D. Spear
License: 
  Software License Agreement (BSD License)

  Copyright 2021 D.S. Watring. All rights reserved.
  Copyright 2021 J. Erickson. All rights reserved.
  Copyright 2021 A.D. Spear. All rights reserved.
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
import math as mt
from numpy import genfromtxt
from numpy import array
from sklearn.neighbors import NearestNeighbors as NN
from matplotlib import pyplot as plt
import os
import glob

#----------------------------------------------------------------------
def import_pores(pore_file):
    #Imports pore file (please see readme for specifics of pore file)
    data = genfromtxt(pore_file, delimiter=',')
    return data

def vdf_function(ahat,bhat,rho,gamma,pores):
    # Pulling the parameter files from the pore csv file
    p = pores[0,0]       #Resolution (in um/pixel)
    L = int(pores[0,1])  #Number of stacks in X-ray CT data (um)
    c = pores[0,2]       #C value from VDF formulation (um)
    V = pores[0,3]       #Total volume of specimen from CT data (um^3)
    
    z_ref = array(range(n+1)) #Z reference array
    # Deletes the above parameters and empyt row of pores file
    pores = np.delete(np.delete(pores, 1, 0),0,0)
    n = len(pores[:,0])  #Number of pores
#----------------------------------------------------------------------
    #Calling parameters
    spher = np.zeros(n)     #Sphericity
    vol = np.zeros(n)       #Volume
    cen = np.zeros([n,3])   #centroids of pores
    r_a = np.zeros(n)       #Semi-major axis
    r_b = np.zeros(n)       #Semi-minor axis
    r_c = np.zeros(n)       #Semi-minor axis
    theta_xy = np.zeros(n)  #Ellipsoid fitting angles
    phi_x = np.zeros(n)     #Ellipsoid fitting angles
    phi_y = np.zeros(n)     #Ellipsoid fitting angles
    ratio = np.zeros(n)     #b/a ratio
    Kt = np.zeros(n)        #Stress concentration term
    value = np.zeros(n)     #VDF values
#----------------------------------------------------------------------
    # Calculates each of the parameters for each pore
    for i in range(n):    
        spher[i] = ((36*mt.pi*(pores[i,3]**2))**(1/3))/pores[i,4] 
        vol[i] = pores[i,3]               #Volume (um^3)
        #For the centroid values, it is in the pore file in pixels
        #Must be converted to um by multiplying by the p value
        cen[i,0] = pores[i,0]*p           #X (um)
        cen[i,1] = pores[i,1]*p           #Y (um)
        cen[i,2] = pores[i,2]*p           #Z (um)
        r_a[i] = pores[i,5]               #Semi-major axis  (um)
        r_b[i] = pores[i,6]               #Semi-minor axis b (um)
        r_c[i] = pores[i,7]               #Semi-minor axis c (um)
        theta_xy[i] = pores[i,8]          #Angles (degrees)
        phi_x[i] = pores[i,10]            #Angles (degrees)
        phi_y[i] = pores[i,9]             #Angles (degrees)
        ratio[i] = r_b[i]/r_a[i]          #b/a ratio 
        kt90 = (ratio[i]**-1)*(2*(90-theta_xy[i])/mt.pi) 
        kt0 = (ratio[i])*(1-2*(90-theta_xy[i])/mt.pi)
        Kt[i] = kt90 - kt0                #Stress concentration
               
    # Calculates the nearest neighbors for each pore    
    nbrs = NN(n_neighbors=n, algorithm='ball_tree').fit(cen)
    distances, _ = nbrs.kneighbors(cen)   #Calculate the distances
    # Deletes first nearest neighbor distance, which is always 0
    distances = np.delete(distances,0,1) 
    # Creates a linear array from 1 to 0
    lin_array = np.linspace(1.0, 0.0, num=n-1)
    # Calculates the weighted distance array
    we_d = np.zeros([len(distances[:,0]),len(distances[0,:])])
    for i in range(len(distances[:,0])):
        for j in range(len(distances[0,:])):
            we_d[i-1,j-1] = distances[i-1,j-1]*lin_array[j-1]

    # Values to normalize nearest neighbor term and scf term        
    all_d = np.max(sum(distances))        #Max sum of distances
    Kt_max = np.max(Kt)                   #Maximum scf  

#----------------------------------------------------------------------
    #VDF Formulation
    vdf_value = np.zeros(len(z_ref))
    for j in range(len(z_ref)): # Calculate for each z_ref value
        for i in range(n): # Calculate for each pore
            #print("i=",i)
            # Calculate the Si term 
            Si = abs(cen[i,2]-z_ref[j]) 
            # Convert angles to radians and calculate for x and y
            anglex = mt.cos(mt.radians(phi_x[i]))
            angley = mt.cos(mt.radians(phi_y[i]))
            anglexy = mt.cos(mt.radians(theta_xy[i]))
            x2 = (cen[i,0] + r_a[i]*anglexy*anglex)**2
            y2 = (cen[i,1] + r_a[i]*anglexy*angley)**2
            
            # Calculate the ri term
            ri = mt.sqrt(x2+y2)
            
            # Calculate each term in VDF            
            s = Si/(ahat*L)               #Si term in VDF
            r = abs(c-ri)/(bhat*c)        #Ri term in VDF 
            # Calculate the nearest neighbor term ai
            ai = (sum(we_d[i,:])/(all_d))*(1/gamma)
            # Calculate the stress concentration term ki
            ki = spher[i]*Kt[i]/(Kt_max*rho) 
            
            # Calculate the VDF value for each pore
            value[i] = vol[i]*np.exp(-s-r-ai-ki)/V

	    # To inspect VDF value with just the nearest neighbor term
	    # Uncomment the following line
            #value[i] = vol[i]*np.exp(-s-r-ai)/V

            # To inspect VDF value with just the stress concentration
	    # Uncomment the following line
            #value[i] = vol[i]*np.exp(-s-r-ki)/V

	# Calculates the VDF value for each z_ref position
        vdf_value[j] = np.nansum(value)
        
    # Plots the VDF value
    plt.plot(z_ref,vdf_value)
    mx = np.argmax(vdf_value) #Location of the maximum VDF value
    # Returns the VDF value values
    return vdf_value
    #return mx #To return the maximum vdf value location


#----------------------------------------------------------------------
if __name__ == "__main__":
    pores = import_pores('example.csv')
    vdf_function(0.342, 1.0, 0.1, 0.1, pores)




