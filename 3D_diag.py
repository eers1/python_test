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

#q_cloud_liquid_mass_average=np.mean(q_cloud_liquid_mass, axis=1)
#q_cloud_liquid_mass_average=np.mean(q_cloud_liquid_mass_average, axis=1)
#q_cloud_liquid_mass_transpose=1e3*(q_cloud_liquid_mass_average.transpose())

Hour_Time=[]
for j in time20600:
        Hour_Time.append((j/3600))

#ax=plt.subplot(111)
#
#x=Hour_Time
#y=z_height
#X, Y = np.meshgrid(x,y)
#
#lower=-0.1
#upper=0.1
#
#levels = np.arange(lower, upper ,0.001)
#
#figure=ax.contourf(X, Y, q_cloud_liquid_mass_transpose, levels=levels)
#ax.set_ylim([0, 120])
#ax.set_xlim([Hour_Time[0], Hour_Time[-1]])
#ax.set_title("Cloud liquid MMR", fontsize=20)
#ax.set_xlabel('Time (UTC)', fontsize=15)
#ax.set_ylabel('Height (m)', fontsize=15)
#
#plt.subplots_adjust(right=0.8, hspace=0.35)
#cax = plt.axes([0.81, 0.1, 0.025, 0.8])
#cbar=plt.colorbar(figure, cax=cax, format='%.2f')
#cbar.set_label("g/kg", labelpad=20, fontsize=15)
#
#plt.show()



#pot_temp_pert_xmean=np.mean(pot_temp_pert, axis=1)
#
#for i in pot_temp_pert_xmean[:][0][0]:
#    print(pot_temp_pert_xmean[i][0][0])
#    pot_temp_pert_xmean_sel=pot_temp_pert_xmean[i][:][:]
#    pot_temp_pert_xmean_transpose=1e3*(pot_temp_pert_xmean_sel.transpose())
#
#    x_mesh=[]
#    for i in range(x_dim):
#        x_mesh.append(i)
#    
#    x=x_mesh
#    y=z_height
#    X, Y = np.meshgrid(x,y)
#
#    lower=pot_temp_pert_xmean_transpose.min()
#    upper=pot_temp_pert_xmean_transpose.max()
#    levels = np.arange(lower, upper, 20)
#
#    plt.figure()
#    ax = plt.subplot(111)
#    cont = ax.contourf(X, Y, pot_temp_pert_xmean_transpose, levels=levels)
#    ax.set_ylabel("Height (m)", fontsize=15)
#    ax.set_xlabel("X direction (m)", fontsize=15)
#
#    plt.title("Pot temp", fontsize=20)
#    plt.legend(loc="best")
#    cbar=plt.colorbar(cont, format='%.2f')
#    cbar.set_label("K", labelpad=20, fontsize=15)
#
#    plt.show()
#    
   
#hori_x_vel_ymean=np.mean(hori_x_vel, axis=1)
#
#lower=hori_x_vel_ymean.min()
#upper=hori_x_vel_ymean.max()
#
#plt.figure(1)
#for i in range(len(hori_x_vel_ymean[:,0,0])):
#    t = time20600[i]
#    hori_x_vel_ymean_sel=hori_x_vel_ymean[i,:,:]
#    hori_x_vel_ymean_transpose=(hori_x_vel_ymean_sel.transpose())
#
#    x_mesh=[]
#    for i in range(x_dim):
#        x_mesh.append(i)
#    
#    x=x_mesh
#    y=z_height
#    X, Y = np.meshgrid(x,y)
#
#    levels = np.arange(lower, upper, 0.01)
#    
#    ax = plt.subplot(111)
#    cont = ax.contourf(X, Y, hori_x_vel_ymean_transpose, label=("t = " + str(i)), levels=levels)
#    ax.set_ylabel("Height (m)", fontsize=15)
#    ax.set_xlabel("X direction (m)", fontsize=15)
#
#    plt.title("Horizontal velocity in y direction", fontsize=20)
#    plt.legend(loc="best")
#    cbar=plt.colorbar(cont, format='%.2f')
#    cbar.set_label("m s$^{-1}$", labelpad=20, fontsize=15)
#
#    plt.show()



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
