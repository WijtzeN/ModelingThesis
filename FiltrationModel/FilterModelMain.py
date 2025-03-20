"""
The main filter model
20/01/2025
Wijtze Nijhuis
"""

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

#simulation parameters
dt = 0.01
t = 300
TotalTime = 0

#process parameters
Phi = 1e-3 #m^3/s
Sl = 0.5e-3 # g/m^3 solid loading
P0 = 1e5+1200 # Pa pressure
dpFib = 1e-6 # Particle size fiber m
dpAdd = 1e-7 # Particle size additive m
Amem = 0.1 # m^2 membrane
dmem = 1e-3 # membrane diameter
Jo = Phi/Amem

#physical properties
Kb = 1 #
Ks = 1 #
Ki = 1 #
Kc = 1 #

#plot spul
v = np.linspace(0,100,100)
dpdvb = (1-Kb*v/Jo)/P0
dpdvs = (1-Ks*v/2)/P0**0.5
dpdvi = np.log(P0)+Ki*v
dpdvc = P0 + Kc*Jo*v

# Plot 1 - Complete blocking
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(v, dpdvb)
plt.ylabel(r"$\frac{1}{P}$", rotation=0, fontsize=20, labelpad=20)
plt.xlabel("v", fontsize=20)
plt.xticks([])
plt.yticks([])
plt.title("Complete Blocking", fontsize=25)

# Plot 2 - Another dp/dv function (using dpdvs)
plt.subplot(2, 2, 2)
plt.plot(v, dpdvs)
plt.ylabel(r"$\frac{1}{\sqrt{P}} $", rotation=0, fontsize=20, labelpad=20)
plt.xlabel("v", fontsize=20)
plt.xticks([])
plt.yticks([])
plt.title("Standard Blocking", fontsize=25)

# Plot 3 - Another dp/dv function (using dpdvi)
plt.subplot(2, 2, 3)
plt.plot(v, dpdvi)
plt.ylabel("ln P", rotation=0, fontsize=20, labelpad=20)
plt.xlabel("v", fontsize=20)
plt.xticks([])
plt.yticks([])
plt.title("Intermediate Blocking", fontsize=25)

# Plot 4 - Constant dp/dv (using dpdvc)
plt.subplot(2, 2, 4)
plt.plot(v, np.full_like(v, dpdvc))
plt.ylabel("P", rotation=0, fontsize=20, labelpad=20)
plt.xlabel("v", fontsize=20)
plt.xticks([])
plt.yticks([])
plt.title("Cake Filtration", fontsize=25)

# Show all plots
plt.tight_layout()

plt.show()

# Lists to store time and v values
times = []
vs = []

# Loop to calculate v and time
while TotalTime < t:
    v = Phi * TotalTime / Amem  # m^3/m^2 volume filtered per unit membrane area
    TotalTime += dt
    times.append(TotalTime)
    vs.append(v)
'''
# Plotting v against time
plt.plot(times, vs, label='Volume Filtered per Unit Membrane Area')
plt.xlabel('Time (s)')
plt.ylabel('v (m^3/m^2)')
plt.title('v vs Time')
plt.legend()
plt.grid(True)
plt.show()
'''
