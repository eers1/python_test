#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:32:17 2019

@author: eers
"""

from netCDF4 import Dataset
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio


# Read in files

file = "fire_sc_trial.nc"

dataset = Dataset(file, "r")

#Assign variables

vert_vel = dataset.variables["w"][:]
hori_x_vel = dataset.variables["u"][:]
hori_y_vel = dataset.variables["v"][:]
pot_temp_pert = dataset.variables["th"][:]
q_water_vap_mmr = dataset.variables["q_vapour"][:]*1000
q_cloud_liquid_mass = dataset.variables["q_cloud_liquid_mass"][:]*1000
time260 = dataset.variables["time_series_2_60.0"][:]
time2600 = dataset.variables["time_series_2_600.0"][:]
time20600 = dataset.variables["time_series_20_600.0"][:]
z_height = dataset.variables["zn"][:]
x_dim = len(dataset.dimensions["x"])
y_dim = len(dataset.dimensions["y"])
dataset.close()

#Hour_Time=[]
#for j in time20600:
#        Hour_Time.append((j/3600))

def plot_single_time( var, horizontal_dimension, z_dimension, time_step, time_series, level_step, Title, xlabel, ylabel, cb_units):
    
    title_string = Title + " t = " + '{:.2f}'.format(time_step)
    
    
    # Average over one horizontal dimension
    if horizontal_dimension == "x":
        horizontal_dimension = x_dim
        var_mean = np.mean(var, axis=1)
    else:
        horizontal_dimension = y_dim
        var_mean = np.mean(var, axis=2)
        
    # Define limits
    lower = var_mean.min()
    upper = var_mean.max()
    
    # Select frame
    T = list(time_series)
    var_mean_ts = var_mean[T.index(time_step),:,:]
    
    # Transpose
    var_mean_ts_T=(var_mean_ts.transpose())#
    #var_mean_ts_T = var_mean_ts_T.astype(float)
    
    # Create grid
    mesh = []
    for i in range(horizontal_dimension):
        mesh.append(i)    
        
    x= mesh
    y= z_dimension
    X, Y = np.meshgrid(x,y)
    
    levels = np.arange(lower, upper, level_step)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #X = X.astype(float)
    contour_plot = ax.contourf(X, Y, var_mean_ts_T, levels=levels)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)

    plt.title(title_string, fontsize=15)
    #plt.legend(loc="best")
    cbar = plt.colorbar(contour_plot, format='%.2f')
    cbar.set_label(cb_units, labelpad=20, fontsize=15)
    
    # IMPORTANT ANIMATION CODE HERE
    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()   
    return image

# Create gifs
kwargs_write = {'fps':1.0, 'quantizer':'nq'}

imageio.mimsave('./vert_velocity.gif', [plot_single_time( vert_vel, "x", z_height, t, time20600, 0.01, "Vertical Velocity", "X direction (m)", "Height (m)", "m s$^{-1}$") for t in time20600], fps=1)

imageio.mimsave('./x_velocity.gif', [plot_single_time( hori_x_vel, "x", z_height, t, time20600, 0.01, "Horizontal velocity in x direction", "X direction (m)", "Height (m)", "m s$^{-1}$") for t in time20600], fps=1)

imageio.mimsave('./y_velocity.gif', [plot_single_time( hori_y_vel, "y", z_height, t, time20600, 0.01, "Horizontal velocity in y direction", "Y direction (m)", "Height (m)", "m s$^{-1}$") for t in time20600], fps=1)

imageio.mimsave('./pot_temp_pert.gif', [plot_single_time( pot_temp_pert, "x", z_height, t, time20600, 0.01, "Potential Temperature Perturbation from Ref", "X direction (m)", "Height (m)", "K") for t in time20600], fps=1)

imageio.mimsave('./vap_mmr.gif', [plot_single_time( q_water_vap_mmr, "x", z_height, t, time20600, 0.01, "Water Vapour Mass Mixing Ratio", "X direction (m)", "Height (m)", "kg kg$^{-1}$") for t in time20600], fps=1)

imageio.mimsave('./cloud_liq_mass.gif', [plot_single_time( q_cloud_liquid_mass, "x", z_height, t, time20600, 0.01, "Cloud Liquid Mass", "X direction (m)", "Height (m)", "kg kg$^{-1}$") for t in time20600], fps=1)
