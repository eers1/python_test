#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:05:58 2019

@author: eers
"""

from netCDF4 import Dataset
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt


# Read in files

file = "fire_sc_trial.nc"

dataset = Dataset(file, "r")

#Assign variables

relative_humidity_mean = dataset.variables["rh_mean"][:]
l_mass_mixing_ratio = dataset.variables["liquid_mmr_mean"][:]*1000
v_mass_mixing_ratio = dataset.variables["vapour_mmr_mean"][:]*1000
pot_temp_mean = dataset.variables["theta_mean"][:]
ww_mean = dataset.variables["ww_mean"][:]
vv_mean = dataset.variables["vv_mean"][:]
uu_mean = dataset.variables["uu_mean"][:]
#vertical_wind_mean = dataset.variables["w_wind_mean"][:]
y_wind_mean = dataset.variables["v_wind_mean"][:]
x_wind_mean = dataset.variables["u_wind_mean"][:]
basic_state_pressure = dataset.variables["prefn"][:]
basic_state_density = dataset.variables["rhon"][:]
ref_pot_temp = dataset.variables["thref"][:]
initial_pot_temp_profile = dataset.variables["thinit"][:]
time = dataset.variables["time_series_2_60.0"][:]
time600 = dataset.variables["time_series_2_600.0"][:]
z_height = dataset.variables["zn"][:]
dataset.close()

Hour_Time=[]
for j in time600:
        Hour_Time.append((j/3600))

X, Y = np.meshgrid(Hour_Time, z_height)

def plot_2d( var, lower, upper, step, title, units, filename ):
    
    levels = np.arange(lower, upper, step)
    var_transpose = np.array(var.transpose())
    
    plt.figure()
    ax = plt.subplot(111)
    cont = ax.contourf(X, Y, var_transpose, levels=levels)
    ax.set_ylabel("Height (m)", fontsize=15)
    ax.set_xlabel("Time (UTC)", fontsize=15)

    plt.title(title, fontsize=20)
#    cont.set_clim(40, 100)
    cbar=plt.colorbar(cont, format='%.2f')
    cbar.set_label(units, labelpad=20, fontsize=15)

    plt.savefig(filename)
    plot = plt.show()
        
    return plot

# Relative Humidity
plot_2d( relative_humidity_mean, 40.0, 101.0, 1.0, "Mean Relative Humidity", "%", "rh_mean" )

# Liquid mass mixing ratio
plot_2d( l_mass_mixing_ratio, 0.0, 0.8, 0.01, "Liquid Mass Mixing Ratio", "%", "l_mmr_f" )

# Vapour mass mixing ratio
plot_2d( v_mass_mixing_ratio, 6.0, 9.5, 0.1, "Vapour Mass Mixing Ratio", "g kg$^{-1}$", "v_mmr_f" )

# Potential temperature mean
plot_2d( pot_temp_mean, 287.0, 301.0, 0.2, "Mean Potential Temperature", "K", "theta_mean_f" )

# Mean wind y direction
plot_2d( y_wind_mean, y_wind_mean.min(), y_wind_mean.max(), 0.002, "Mean wind in y component", "m s$^{-1}$", "v_wind_f" )

# Mean wind x direction
plot_2d( x_wind_mean, x_wind_mean.min(), x_wind_mean.max(), 0.01, "Mean wind in x direction", "m s$^{-1}$", "u_wind_f" )

# vv_mean
plot_2d( vv_mean, vv_mean.min(), vv_mean.max(), 0.002, "vv mean", "m$^{2}$ s$^{-2}$", "vv_wind_f" )

# uu mean
plot_2d( uu_mean, uu_mean.min(), uu_mean.max(), 0.002, "uu mean", "m$^{2}$ s$^{-2}$", "uu_wind_f" )

# ww mean
plot_2d( ww_mean, ww_mean.min(), ww_mean.max(), 0.002, "ww_mean", "m$^{2}$ s$^{-2}$", "ww_wind_f" )

# Basic State Pressure
plot_2d( basic_state_pressure, basic_state_pressure.min(), basic_state_pressure.max(), 200, "Basic State Pressure", "Pa", "prefn_f" )

# Basic State Density
plot_2d( basic_state_density, 0, 1.2, 0.1, "Basic State Pressure", "Pa", "prefn_f" )

# Reference Potential Temperature
plot_2d( ref_pot_temp, 280, 300, 1, "Reference Potential Temperature", "K", "thref_f" )

# Initial Potential Temperature Profile
plot_2d( initial_pot_temp_profile, initial_pot_temp_profile.min() - 1, initial_pot_temp_profile.max() + 1, 0.5, "Initial Potential Temperature Profile", "K", "thinit_f" )
