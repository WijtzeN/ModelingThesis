import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from keras.src.layers import average
from scipy.optimize import curve_fit
from scipy.stats import t
from scipy.signal import savgol_filter
from CoolProp.CoolProp import PropsSI
import FunctionFile as FF
import mplcursors
import statsmodels.api as sm
#matplotlib.use('TkAgg')

# --- Data ---
file_name = 'Run11.csv'
df = pd.read_csv(file_name)
pre_p = df["Pre filter Pressure Gem. (mbar)"].fillna(0)
post_p = df["Post filter Pressure Gem. (mbar)"].fillna(0)
P_data = (pre_p - post_p) * 100  # Convert mbar to Pa
P_data = P_data.reset_index(drop=True)
time_data = np.arange(1, len(P_data) + 1)
#time_data = df["time"].values
Temp_data = df["post filter Gem. (C)"].fillna(0).values + 273.15301
# --- Limits for plotting ---
t_flat_start = 0
t_flat_end = 1
t_rise_start = t_flat_end
t_rise_end = t_rise_start + 1000
t_clean = 1680

x_bounds = [000,3400]
y_bounds = (-400, 1500)
base_bounds = (00, 50)
correction_P_data = np.mean(P_data[base_bounds[0]:base_bounds[1]])
P_data = P_data - correction_P_data  # Correct pressure data

#P_data_smooth = P_data.ewm(alpha=0.1).mean()
P_data_smooth = savgol_filter(P_data, window_length=25, polyorder=1)

# --- Process parameters ---
#Temp = 200 + 273.15  # K replaced with pico data
Q_water = 0  # mL/min
Q_water_m3 = Q_water / 1000 / 1000 / 60  # Convert to m³/s
Q_N2 = 10  # L/min
M = 0  # Dust mass in g
Substance = 'Cellulose'  # 'CaCO3' or 'Cellulose'
#l_final = (M / 1000) / (rho_Cell * A)  # Final cake thickness (m)

# --- Constants ---
A = 0.25 * (6 * 0.01)**2 * np.pi  # Membrane area (m²)
#e_void = 0.55  # Void fraction
ref_temp = 20 + 273.15  # Reference temperature in Kelvin

# --- Physical properties ---
MM_H2O = 18.01528  # g/mol
R = 8.314  # J/(mol*K)
MM_N2 = 28.0134  # g/mol
Patm = 105137  # Pa
rho_CaCO3 = 0.546675*1000##0.28*1000#1500#197.8  # kg/m³
rho_Cell = 0.2796*1000  # kg/m³
rho_water = PropsSI("D", "T", ref_temp, "P", Patm, "Water")


# --- Process parameter calculations ---
Q_steam_time = []
Q_N2_m3_only = []
mu_time = []

T_ref = ref_temp  # 293.15 K
rho_N2_ref = PropsSI("D", "T", T_ref, "P", Patm, "Nitrogen")  # kg/m³
Q_N2_m3s_ref = Q_N2 / 1000 / 60  # m³/s
if Substance == 'CaCO3':
    rho_Substance = rho_CaCO3
elif Substance == 'Cellulose':
    rho_Substance = rho_Cell
else:
    raise ValueError("Substance must be either 'CaCO3' or 'Cellulose'")

mdot_N2 = Q_N2_m3s_ref * rho_N2_ref  # constant mass flow of N2
i = 1
for T in Temp_data:
    # --- Steam flow ---
    mdot_water = Q_water_m3 * rho_water
    i+=1
    D_steam = PropsSI("D", "T", T, "P", Patm, "Water")  # m³/kg
    if i == 7700:
        print(f"Density at {T} K: {D_steam} kg/m³")

    Q_steam_only = mdot_water / D_steam

    # --- Nitrogen flow ---
    rho_N2_actual = PropsSI("D", "T", T, "P", Patm, "Nitrogen")
    Q_N2_m3 = mdot_N2 / rho_N2_actual

    # --- Total flow and viscosity ---
    Q_total = Q_steam_only + Q_N2_m3
    Q_N2_m3_only.append(Q_N2_m3)
    Q_steam_time.append(Q_total)
    #Q_N2_m3 = Q_N2 / 1000 / 60 / ref_temp * T
    #Q_steam = Q_water_m3 * rho_water * 1000 / MM_H2O * (R * T / Patm) + Q_N2_m3
    #Q_steam_time.append(Q_steam)

    mu_time.append(FF.musteam(T, Q_N2_m3 / Q_total))
# --- Convert lists to arrays ---
time_data = np.array(time_data)
P_data = np.array(P_data)
Q_steam_time = np.array(Q_steam_time)
Q_N2_m3_only = np.array(Q_N2_m3_only)
mu_time = np.array(mu_time)

print(f'Q at clean time: {Q_steam_time[t_clean]} m³/s')
print(f'mu at clean time: {mu_time[t_clean]} Pa s')
# --- Extract flat and rise data ---
mask_flat = (time_data >= t_flat_start) & (time_data <= t_flat_end)
mask_rise = (time_data >= t_rise_start) & (time_data <= t_rise_end)

t_flat = time_data[mask_flat]
P_flat = P_data[mask_flat]
mu_flat = mu_time[mask_flat]
Q_flat = Q_steam_time[mask_flat]
Q_N2_flat = Q_N2_m3_only[mask_flat]
Temp_flat = Temp_data[mask_flat]

print(f'Q_flat: {np.mean(Q_flat)} m³/s')

t_rise = time_data[mask_rise]
P_rise = P_data[mask_rise]
mu_rise = mu_time[mask_rise]
Q_rise = Q_steam_time[mask_rise]
#l_rise = np.linspace(0, l_final, len(t_rise))

Temp_flat = np.mean(Temp_flat)
mdot_water = Q_water_m3 * rho_water
mdot_total = mdot_water + mdot_N2     # ≈ (9.970e-5 + 7.767e-5) = 1.774e-4 kg/s
w_steam = mdot_water / mdot_total  # ≈ 9.970e-5 / 1.774e-4 ≈ 0.562
w_N2 = mdot_N2 / mdot_total        # ≈ 7.767e-5 / 1.774e-4 ≈ 0.438
rho_steam_op = PropsSI("D", "T", Temp_flat, "P", (Patm + np.mean(P_flat)), "water")    # ≈ 0.460 kg/m³
rho_N2_op = PropsSI("D", "T", Temp_flat, "P", (Patm + np.mean(P_flat)), "Nitrogen")    # ≈ 0.742 kg/m³
rho_mix_flat = 1 / (w_N2/rho_N2_op + w_steam/rho_steam_op)
print(f'mass N2: {mdot_N2} kg/s, mass steam: {mdot_water} kg/s, total mass: {mdot_total} kg/s')
print(f'density steam: {rho_steam_op} kg/m³, density N2: {rho_N2_op} kg/m³, density mix: {rho_mix_flat} kg/m³')

#values for Forchheimer
print(f'ΔP (Pa): {np.mean(P_flat)}',
      f'μ_N2 (Pa s): {PropsSI("V", "T", Temp_flat, "P", Patm, "Nitrogen")}',
      f'μ (Pa s): {np.mean(mu_flat)}',
      f'Temp (K): {Temp_flat}',
      f'J (m/s): {np.mean(Q_flat)/A}',
      f'J_N2 (m/s): {np.mean(Q_N2_flat)/A}',
      f'rho_N2 (kg/m³): {PropsSI("D", "T", Temp_flat, "P", Patm, "Nitrogen")}',
      f'rho_mix (kg/m³): {rho_mix_flat}',
      sep='\n')

max_P_smooth_rise = P_data_smooth[mask_rise].max()
print(f"Maximum value of P_smooth in the mask_rise time: {max_P_smooth_rise}")
# Find the index of the maximum value in P_smooth within mask_rise
idx_max_rise = np.argmax(P_data_smooth[mask_rise])

# Convert the local index to the global index in the full arrays
idx_global = np.where(mask_rise)[0][0] + idx_max_rise

# Extract the time point, Q_rise, and mu_rise values
time_peak = time_data[idx_global]
Q_peak = Q_steam_time[idx_global]
mu_peak = mu_time[idx_global]

R_peak = max_P_smooth_rise * A  / (Q_peak * mu_peak)  # Membrane resistance at peak
print(f"R_peak in the mask_rise time: {R_peak:.3e} m⁻¹")


# --- Fitting membrane resistance ---
def R_mem_fit(x, R_mem):
    return x * R_mem

x_flat = Q_flat * mu_flat / A
popt_Rmem, pcov_Rmem = curve_fit(R_mem_fit, x_flat, P_flat)
R_mem = popt_Rmem[0]
P_fit_flat = R_mem * x_flat
P_min = float(min(P_fit_flat))

# --- Stats for membrane fit ---
residuals_flat = P_flat - P_fit_flat
ss_res_flat = np.sum(residuals_flat**2)
ss_tot_flat = np.sum((P_flat - np.mean(P_flat))**2)
rmse_flat = np.sqrt(np.mean(residuals_flat**2))

max_P_smooth_rise_dif = P_data_smooth[mask_rise].max() - P_min
print(f"P increase in in the mask_rise time: {max_P_smooth_rise_dif}")

idx_max_rise = np.argmax(P_data_smooth[mask_rise])
idx_global = np.where(mask_rise)[0][0] + idx_max_rise  # actual index in full arrays

# --- Calculate alpha ---
alpha_val = (max_P_smooth_rise_dif * rho_Cell * A**2) / (Q_peak * mu_peak * M / 1000)  # M in grams -> kg
print(f"Alpha at pressure peak: {alpha_val:.2e} m^-2")

alpha = 0.05
dof_flat = len(P_flat) - 1
tval_flat = t.ppf(1 - alpha/2, dof_flat)
perr_Rmem = np.sqrt(np.diag(pcov_Rmem))
R_mem_ci = tval_flat * perr_Rmem[0]
#
# # --- Fitting cake resistance ---
# def r_fit(xdata, r):
#     Q_rise, mu_rise, l_rise = xdata
#     return Q_rise * mu_rise * r * l_rise / A + P_min
#
# xdata_r = (Q_rise, mu_rise, l_rise)
# popt_r, pcov_r = curve_fit(r_fit, xdata_r, P_rise, bounds=([1e5], [1e20]))
# r = popt_r[0]
# P_fit_rise = r_fit(xdata_r, r)

# # --- Stats for cake resistance fit ---
# residuals_rise = P_rise - P_fit_rise
# ss_res_rise = np.sum(residuals_rise**2)
# ss_tot_rise = np.sum((P_rise - np.mean(P_rise))**2)
# r2_rise = 1 - (ss_res_rise / ss_tot_rise)
# rmse_rise = np.sqrt(np.mean(residuals_rise**2))
#
# dof_r = len(P_rise) - 1
# tval_r = t.ppf(1 - alpha/2, dof_r)
# perr_r = np.sqrt(np.diag(pcov_r))
# r_ci = tval_r * perr_r[0]

# --- Confidence bounds for cake resistance ---
# r_upper = r + r_ci
# r_lower = r - r_ci
# P_fit_rise_upper = r_fit(xdata_r, r_upper)
# P_fit_rise_lower = r_fit(xdata_r, r_lower)

plt.figure(figsize=(10, 6))
plt.grid(True, which='major', linewidth=0.8)
plt.minorticks_on()
plt.gca().yaxis.set_minor_locator(MultipleLocator(100))
plt.grid(True, which='minor', linestyle=':', linewidth=0.5)

# Raw and smoothed data
sc = plt.scatter(time_data, P_data, label='Raw data', alpha=0.4)
plt.plot(time_data, P_data_smooth, label='Smoothed data', color='blue', linewidth=1)

# Membrane resistance fit
plt.plot(t_flat, P_fit_flat,
         label=fr'Fit: $R_{{mem}}$ = {R_mem:.2e} ± {R_mem_ci:.1e} m⁻¹, RMSE = {rmse_flat:.1f} Pa',
         color='orange', linewidth=2)

# Labels and title
plt.ylabel(r"$\Delta P$ (Pa)", rotation=90, fontsize=15, labelpad=20)
plt.xlabel("Time (s)", fontsize=15)
#plt.title(f"Cake Filtration Fit Q = {Q_water}ml/m, N2 = {Q_N2} l/m, run {file_name}", fontsize=20)
#plt.legend(loc='best', fontsize=10)
plt.xlim(x_bounds)
plt.ylim(y_bounds)
plt.tight_layout()

# ADD Hover functionality
mplcursors.cursor(sc, hover=True)

plt.show()