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

def plot_2d( var, lower, upper, step, title, units, file ):
    
    levels = np.arange(lower, upper, step)
    transpose = np.array(var.transpose())
    
    plt.figure()
    ax = plt.subplot(111)
    cont = ax.contourf(X, Y, transpose, levels=levels)
    ax.set_ylabel("Height (m)", fontsize=15)
    ax.set_xlabel("Time (UTC)", fontsize=15)

    plt.title(title, fontsize=20)
    cont.set_clim(40, 100)
    cbar=plt.colorbar(cont, format='%.2f')
    cbar.set_label(units, labelpad=20, fontsize=15)

    plt.savefig(file)
    plot = plt.show()
    
    return plot

# Relative Humidity
plot_2d( relative_humidity_mean, 40.0, 101.0, 1.0, "Mean Relative Humidity", "%", "rh__mean" )

# Liquid mass mixing ratio
plot_2d( l_mass_mixing_ratio, 0.0, 0.8, 0.01, "Liquid Mass Mixing Ratio", "%", "l_mmr_f" )

lower=0.0
upper=0.8
levels = np.arange(lower, upper, 0.01)

l_mass_mixing_ratio_g = l_mass_mixing_ratio.transpose()*1000

plt.figure()
ax = plt.subplot(111)
cont = ax.contourf(X, Y, l_mass_mixing_ratio_g, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Liquid Mass Mixing Ratio", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("%", labelpad=20, fontsize=15)

plt.savefig("l_mmr")
plt.show()

# Vapour mass mixing ratio
plot_2d( v_mass_mixing_ratio, 6.0, 9.5, 0.1, "Vapour Mass Mixing Ratio", "%", "v_mmr_f" )

lower=6.0
upper=9.5
levels = np.arange(lower, upper, 0.1)

v_mass_mixing_ratio_g = v_mass_mixing_ratio.transpose()*1000

plt.figure()
ax = plt.subplot(111)
cont = ax.contourf(X, Y, v_mass_mixing_ratio_g, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Vapour Mass Mixing Ratio", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("g kg$^{-1}$", labelpad=20, fontsize=15)

plt.savefig("v_mmr")
plt.show()

# Potential temperature mean
plot_2d( pot_temp_mean, 287.0, 301.0, 0.2, "Mean Potential Temperature", "K", "theta_mean_f" )

lower=287.0
upper=301.0
levels = np.arange(lower, upper, 0.2)

pot_temp_mean_T = pot_temp_mean.transpose()

plt.figure()
ax = plt.subplot(111)
cont = ax.contourf(X, Y, pot_temp_mean_T, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Mean Potential Temperature", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("K", labelpad=20, fontsize=15)

plt.savefig("theta_mean")
plt.show()

# Mean wind y direction
plot_2d( y_wind_mean, y_wind_mean.mix(), y_wind_mean.max(), 0.002, "Mean wind in y component", "m s$^{-1}$", "v_wind_f" )

y_wind_mean_T = y_wind_mean.transpose()

lower=y_wind_mean_T.min()
upper=y_wind_mean_T.max()
levels = np.arange(lower, upper, 0.002)

plt.figure(5)
ax = plt.subplot(111)
cont = ax.contourf(X, Y, y_wind_mean_T, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Mean wind in y component", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("m s$^{-1}$", labelpad=20, fontsize=15)

plt.savefig("v_wind")
plt.show()

# Mean wind x direction
plot_2d( x_wind_mean, x_wind_mean.mix(), x_wind_mean.max(), 0.01, "Mean wind in x direction", "m s$^{-1}$", "u_wind_f" )

x_wind_mean_T = np.array(x_wind_mean.transpose())

lower=x_wind_mean_T.min()  # need to fix this
upper=x_wind_mean_T.max()
levels = np.arange(lower, upper, 0.01)

plt.figure()
ax = plt.subplot(111)
cont = ax.contourf(X, Y, x_wind_mean_T, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Mean wind in x direction", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("m s$^{-1}$", labelpad=20, fontsize=15)

plt.savefig("u_wind")
plt.show()

# vv_mean
plot_2d( vv_mean, vv_mean.mix(), vv_mean.max(), 0.002, "vv mean", "m$^{2}$ s$^{-2}$", "vv_wind_f" )

vv_mean_T = vv_mean.transpose()

lower=vv_mean_T.min()
upper=vv_mean_T.max()
levels = np.arange(lower, upper, 0.002)

plt.figure()
ax = plt.subplot(111)
cont = ax.contourf(X, Y, vv_mean_T, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("vv mean", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("m$^{2}$ s$^{-2}$", labelpad=20, fontsize=15)

plt.savefig("vv_mean")
plt.show()

# uu mean
plot_2d( uu_mean, uu_mean.mix(), uu_mean.max(), 0.002, "uu mean", "m$^{2}$ s$^{-2}$", "uu_wind_f" )

uu_mean_T = np.array(uu_mean.transpose())

lower=uu_mean_T.min()  
upper=uu_mean_T.max()
levels = np.arange(lower, upper, 0.002)

plt.figure()
ax = plt.subplot(111)
cont = ax.contourf(X, Y, uu_mean_T, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("uu mean", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("m$^{2}$ s$^{-2}$", labelpad=20, fontsize=15)

plt.savefig("uu_mean")
plt.show()

# ww mean
plot_2d( ww_mean, ww_mean.mix(), ww_mean.max(), 0.002, "ww_mean", "m$^{2}$ s$^{-2}$", "ww_wind_f" )

ww_mean_T = np.array(ww_mean.transpose())

lower=ww_mean_T.min() 
upper=ww_mean_T.max()
levels = np.arange(lower, upper, 0.002)

plt.figure()
ax = plt.subplot(111)
cont = ax.contourf(X, Y, ww_mean_T, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("ww mean", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("m$^{2}$ s$^{-2}$", labelpad=20, fontsize=15)

plt.savefig("ww_mean")
plt.show()

# Basic State Pressure
plot_2d( basic_state_pressure, basic_state_pressure.mix(), basic_state_pressure.max(), 200, "Basic State Pressure", "Pa", "prefn_f" )

basic_state_pressure_T = np.array(basic_state_pressure.transpose())

lower=basic_state_pressure_T.min()
upper=basic_state_pressure_T.max()
levels = np.arange(lower, upper, 200)

plt.figure()
ax = plt.subplot(111)
cont = ax.contourf(X, Y, basic_state_pressure_T, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Basic State Pressure", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("Pa", labelpad=20, fontsize=15)

plt.savefig("prefn")
plt.show()

# Basic State Density
plot_2d( basic_state_density, 0, 1.2, 0.1, "Basic State Pressure", "Pa", "prefn_f" )

basic_state_density_T = np.array(basic_state_density.transpose())

lower=0
upper=1.2
levels = np.arange(lower, upper, 0.1)

plt.figure()
ax = plt.subplot(111)
cont = ax.contourf(X, Y, basic_state_density_T, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Basic State Density", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("kg m$^{-3}$", labelpad=20, fontsize=15)

plt.savefig("rhon")
plt.show()

# Reference Potential Temperature
plot_2d( ref_pot_temp, 280, 300, 1, "Reference Potential Temperature", "K", "thref_f" )

ref_pot_temp_T = np.array(ref_pot_temp.transpose())

lower=280
upper=300
levels = np.arange(lower, upper, 1)

plt.figure()
ax = plt.subplot(111)
cont = ax.contourf(X, Y, ref_pot_temp_T, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Reference Potential Temperature", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("K", labelpad=20, fontsize=15)

plt.savefig("thref")
plt.show()

# Initial Potential Temperature Profile
plot_2d( initial_pot_temp_profile, initial_pot_temp_profile.min() - 1, initial_pot_temp_profile.max() + 1, 0.5, "Initial Potential Temperature Profile", "K", "thinit_f" )

initial_pot_temp_profile_T = np.array(initial_pot_temp_profile.transpose())

lower=initial_pot_temp_profile_T.min() - 1
upper=initial_pot_temp_profile_T.max() + 1
levels = np.arange(lower, upper, 0.5)

plt.figure()
ax = plt.subplot(111)
cont = ax.contourf(X, Y, initial_pot_temp_profile_T, levels=levels)
ax.set_ylabel("Height (m)", fontsize=15)
ax.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Initial Potential Temperature Profile", fontsize=20)
cbar=plt.colorbar(cont, format='%.2f')
cbar.set_label("K", labelpad=20, fontsize=15)

plt.savefig("thinit")
plt.show()