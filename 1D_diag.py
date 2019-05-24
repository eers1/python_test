from netCDF4 import Dataset
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

# Read in files

file = "diagnostics_ts_10800.0.nc"

dataset = Dataset(file, "r")

# Assign variables - 1D

LWP = dataset.variables["LWP_mean"][:]
VWP = dataset.variables["VWP_mean"][:]
qlmax = dataset.variables["qlmax"][:]
cltop = dataset.variables["cltop_mean"][:]
clbas = dataset.variables["clbas_mean"][:]
cltop_max = dataset.variables["cltop_max"][:]
clbas_max = dataset.variables["clbas_max"][:]
cltop_min = dataset.variables["cltop_min"][:]
wmax = dataset.variables["wmax"][:]
wmin = dataset.variables["wmin"][:]
time = dataset.variables["time_series_2_60.0"][:]
time600 = dataset.variables["time_series_2_600.0"][:]
z_height = dataset.variables["zn"][:]

# Assign Dimensions

x = dataset.dimensions["x"]
y = dataset.dimensions["y"]
z = dataset.dimensions["z"]

dataset.close()

# Plot LWP 

plt.figure()
ax1 = plt.subplot(111)
ax1.plot(time, LWP*1000, label="LWP", linewidth=2)
ax1.set_ylabel("Liquid Water Path (g m$^{-2}$)", fontsize=15)
ax1.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Liquid Water Path")
plt.legend(loc="best")
plt.savefig("LWP")

plt.figure()
ax2 = plt.subplot(111)
ax2.plot(time, cltop, label="Cloud top mean", linewidth=2)
ax2.plot(time, clbas, label="Cloud base mean", linewidth=2)
ax2.set_ylabel("Height (m)", fontsize=15)
ax2.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Cloud Height Mean", fontsize=15)
plt.legend(loc="best")
plt.savefig("Cloud_height")
plt.show()

#plt.figure(3)
#ax3 = plt.subplot(111)
#ax3.plot(time, cltop_max, label="Cloud top max", linewidth=2)
#ax3.plot(time, clbas_max, label="Cloud base max", linewidth=2)
#ax3.set_ylabel("Height (m)", fontsize=15)
#ax3.set_xlabel("Time (UTC)", fontsize=15)
#
#plt.title("Cloud Max", fontsize=15)
#plt.legend(loc="best")
#plt.show()
#
#plt.figure(4)
#ax4 = plt.subplot(111)
#ax4.plot(time, cltop, label="Cloud top mean", linewidth=2)
#ax4.plot(time, cltop_max, label="Cloud top max", linewidth=2)
#ax4.plot(time, cltop_min, label="Cloud top min", linewidth=2)
#ax4.set_ylabel("Height (m)", fontsize=15)
#ax4.set_xlabel("Time (UTC)", fontsize=15)
#
#plt.title("Cloud Top", fontsize=15)
#plt.legend(loc="best")
#plt.show()

plt.figure()
ax5 = plt.subplot(111)
ax5.plot(time, VWP, label="VWP", linewidth=2)
ax5.set_ylabel("Liquid Water Path (g m$^{-2}$)", fontsize=15)
ax5.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Vapour Water Path", fontsize=15)
plt.legend(loc="best")
plt.savefig("VWP")
plt.show()

plt.figure()
ax6 = plt.subplot(111)
ax6.plot(time, qlmax, label="ql max", linewidth=2)
ax6.set_ylabel("Height (m)", fontsize=15)
ax6.set_xlabel("Time (UTC)", fontsize=15)

plt.title("ql max", fontsize=15)
plt.legend(loc="best")
plt.savefig("ql_max")
plt.show()

plt.figure()
ax2 = plt.subplot(111)
ax2.plot(time, wmax, label="W max", linewidth=2)
ax2.plot(time, wmin, label="W min", linewidth=2)
ax2.set_ylabel("Vertical velocity (m s$^{-1}$)", fontsize=15)
ax2.set_xlabel("Time (UTC)", fontsize=15)

plt.title("Vertical Velocity, W", fontsize=15)
plt.legend(loc="best")
plt.savefig("W_maxmin")
plt.show()