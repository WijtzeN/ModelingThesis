import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
import pandas as pd
import os
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker

# Folder for saving plots
os.makedirs("PlotsOptimisation", exist_ok=True)


# Base variable definitions
FinalWeight = 1000  # kg/h
WaterConcFinal = 0.05
WaterConcInit = 0.5
InitTempPaper = 60 + 273.15
FinalTempPaper = 150 + 273.15
C_foulant = 1e-3  # g/m³
years = 6
FinalTime = 8600 * years  # h
priceM2_base = 80  # €/m²

# Base parameters
Area_base = 200
Area_min = 10
Area_max = 500
AreaSteps = int((Area_max - Area_min)/2) + 1

HoursCleaning_base = 4300
HoursCleaning_min = 100
HoursCleaning_max = 8600*6+1
HoursCleaningStepSize = 200


InitTempSteam_base = 291 + 273.15
InitTempSteam_min = 250 + 273.15
InitTempSteam_max = 350 + 273.15
InitTempSteamStepSize = 2

FinalTempSteam_base = 178 + 273.15
FinalTempSteam_min = 160 + 273.15
FinalTempSteam_max = 200 + 273.15
FinalTempSteamStepSize = 2


# Backwash Parameters
BackwashFlux = 0.1  # m/s
InletWaterTemp = 25 + 273.15
BackwashSteamTemp = 140 + 273.15
FilterFractionCleaned = 0.1
ActiveBackwashTime = 2/60     # hours
TotalBackwashTime = 1         # hours (does not affect energy)

# Scan ranges
HoursCleaning_vals = np.arange(HoursCleaning_min, HoursCleaning_max + HoursCleaningStepSize, HoursCleaningStepSize)
Area_vals = np.linspace(Area_min, Area_max, AreaSteps)
InitTempSteam_vals = np.arange(InitTempSteam_min, InitTempSteam_max + InitTempSteamStepSize, InitTempSteamStepSize)
FinalTempSteam_vals = np.arange(FinalTempSteam_min, FinalTempSteam_max + FinalTempSteamStepSize, FinalTempSteamStepSize)

# Energy prices €/kWh
prices = {
    "NL": 0.2046,
    "Europe": 0.1902,
    "Solar": 0.037,
    "Wind": 0.029
    }

# Total energy calculation
def calculate_total_energy(area=Area_base, HoursCleaning=HoursCleaning_base, init_steam=InitTempSteam_base, FinalTempSteam=FinalTempSteam_base):
    hPerClean = HoursCleaning
    time_steps = FinalTime
    time = np.linspace(0, FinalTime, time_steps)

    # Water calculations
    initial_water_weight = WaterConcInit * FinalWeight * (1 - WaterConcFinal) / (1 - WaterConcInit)
    WaterRemoved = initial_water_weight - FinalWeight * WaterConcFinal

    # Energy to heat paper
    cp_paper = 1260  # J/kg/K
    Q_paper = FinalWeight * cp_paper * (FinalTempPaper - InitTempPaper)

    # Energy to evaporate water
    h_water_in = PropsSI('H', 'T', InitTempPaper, 'P', 101325, 'Water')
    h_steam_out = PropsSI('H', 'T', FinalTempSteam, 'P', 101325, 'Water')
    Q_water = WaterRemoved * (h_steam_out - h_water_in)

    # Steam mass flow required
    h_steam_in = PropsSI('H', 'T', init_steam, 'P', 101325, 'Water')
    Q_per_kg_steam = h_steam_in - h_steam_out
    steam_mass_flow = (Q_paper + Q_water) / Q_per_kg_steam  # kg/h

    total_steam_mass = steam_mass_flow + WaterRemoved

    # Steam properties
    rho_steam = PropsSI('D', 'T', FinalTempSteam, 'P', 101325, 'Water')
    mu_steam = PropsSI('V', 'T', FinalTempSteam, 'P', 101325, 'Water')

    volumetric_flow = (total_steam_mass / 3600) / rho_steam  # m³/s

    # Backwash calculations
    h_water_inlet_backwash = PropsSI('H', 'T', InletWaterTemp, 'P', 101325, 'Water')
    h_steam_backwash = PropsSI('H', 'T', BackwashSteamTemp, 'P', 101325, 'Water')
    rho_steam_backwash = PropsSI('D', 'T', BackwashSteamTemp, 'P', 101325, 'Water')

    # Total energy over lifetime
    TotalEnergy_kWh = 0
    IrDust = 0
    TotalEnergyActiveBackwash = 0
    counter =0
    pressure_values = []
    for i, t in enumerate(time[:-3]):
        time_since_clean = (t % hPerClean)
        # Run backwash once per cycle

        if t >= hPerClean and (t % hPerClean) < 1:
            IrDust = (1-0.96) * C_foulant * volumetric_flow * 3600*hPerClean

            # Active backwash steam energy
            energy_active_backwash = (rho_steam_backwash * BackwashFlux * area *
                                     (h_steam_backwash - h_water_inlet_backwash)) * ActiveBackwashTime #Wh

            energy_active_backwash /= 1e3  # kWh
            TotalEnergyActiveBackwash += energy_active_backwash
            TotalEnergy_kWh += energy_active_backwash
            #print("hi")

        # Fouling pressure + motor energy
        if t >= hPerClean and (t % hPerClean) < 1:
            J = volumetric_flow / (area * (1 - FilterFractionCleaned))
            M_t = C_foulant * volumetric_flow * hPerClean/2 * 3600 + IrDust
        else:
            J = volumetric_flow / area
            M_t = C_foulant * volumetric_flow * time_since_clean * 3600 + IrDust


        P0 = (1.464e8 + (J * mu_steam)**2 * FinalTempSteam) * J * mu_steam
        Pt = P0 + 8.188e8 * (J * mu_steam)**2 * FinalTempSteam * M_t / area
        pressure_values.append(Pt)
        E_kW = Pt * volumetric_flow / 1000

        if i > 0:
            dt = time[i] - time[i - 1]
            TotalEnergy_kWh += 0.5 * (E_kW + prev_E_kW) * dt

        prev_E_kW = E_kW
    return TotalEnergy_kWh,  np.array(pressure_values)

# -1 Init Steam plot
print("Calculating Init Steam Temperature sensitivity...")
TotalEnergy_vals = []

for cy in InitTempSteam_vals:
    E, P = calculate_total_energy(init_steam=cy)
    TotalEnergy_vals.append(E)

plt.figure(figsize=(10, 6))
ax1 = plt.gca()

ax1.plot(InitTempSteam_vals-273.15, TotalEnergy_vals, label="Energy use", color="blue")
ax1.set_yscale("log")
ax1.set_xlabel("Initial steam temperature, before drying (°C)")
ax1.set_ylabel("Total energy (kWh)", color="k")
ax1.tick_params(axis='y', labelcolor='k')
ax1.grid(True, which="both", ls="--")

ax2 = ax1.twinx()
ax2.set_yscale("log")
ax2.set_ylabel("Total cost (€)", color="k")
ax2.grid(False)

results_initTemp = []

for key, price in prices.items():
    costs = np.array(TotalEnergy_vals) * price + Area_base * priceM2_base
    ax2.plot(InitTempSteam_vals-273.15, costs, '--', label=f"Total cost ({key})")
    results_initTemp.append(costs)

df_initTemp = pd.DataFrame(
    np.column_stack([InitTempSteam_vals] + results_initTemp),
    columns=["initTemp"] + list(prices.keys())
)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

#plt.title("Sensitivity of initial Steam Temperature")
plt.tight_layout()
plt.savefig("PlotsOptimisation/InitTemp.png")
plt.show()

# 0. Final temp plot
print("Calculating Final Steam Temperature sensitivity...")
TotalEnergy_vals = []
TempDif=FinalTempSteam_base - InitTempSteam_base
for cy in FinalTempSteam_vals:
    E, P = calculate_total_energy(FinalTempSteam=cy, init_steam=cy - TempDif)
    TotalEnergy_vals.append(E)

plt.figure(figsize=(10, 6))
ax1 = plt.gca()

ax1.plot(FinalTempSteam_vals-273.15, TotalEnergy_vals, label="Energy use", color="blue")
ax1.set_yscale("log")
ax1.set_xlabel("Final temperature after drying (°C)")
ax1.set_ylabel("Total energy (kWh)", color="k")
ax1.tick_params(axis='y', labelcolor='k')
ax1.grid(True, which="both", ls="--")

ax2 = ax1.twinx()
ax2.set_yscale("log")
ax2.set_ylabel("Total cost (€)", color="k")
ax2.grid(False)

results_FinalTemp = []

for key, price in prices.items():
    costs = np.array(TotalEnergy_vals) * price + Area_base * priceM2_base
    ax2.plot(FinalTempSteam_vals-273.15, costs, '--', label=f"Total cost ({key})")
    results_FinalTemp.append(costs)

df_FinalTemp = pd.DataFrame(
    np.column_stack([FinalTempSteam_vals] + results_FinalTemp),
    columns=["FinalTemp"] + list(prices.keys())
)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

#plt.title("Sensitivity of Final Steam Temperature")
plt.tight_layout()
plt.savefig("PlotsOptimisation/FinalTemp.png")
plt.show()


# 1. HoursCleaning plot
print("Calculating Hours between cleanings sensitivity...")
TotalEnergy_vals = []
HoursCleaning_vals = np.array([h for h in HoursCleaning_vals if FinalTime % h == 0])
print(HoursCleaning_vals)
for cy in HoursCleaning_vals:
    E, P = calculate_total_energy(HoursCleaning=cy)
    TotalEnergy_vals.append(E)

plt.figure(figsize=(10, 6))
ax1 = plt.gca()

ax1.plot(HoursCleaning_vals, TotalEnergy_vals, label="Energy use", color="blue")
ax1.set_yscale("log")
ax1.set_xlabel("Hours in between cleaning (h)")
ax1.set_ylabel("Total energy (kWh)", color="k")
ax1.tick_params(axis='y', labelcolor='k')
ax1.grid(True, which="both", ls="--")

ax2 = ax1.twinx()
ax2.set_yscale("log")
ax2.set_ylabel("Total cost (€)", color="k")
ax2.grid(False)

results_clean = []

for key, price in prices.items():
    costs = np.array(TotalEnergy_vals) * price + Area_base * priceM2_base
    ax2.plot(HoursCleaning_vals, costs, '--', label=f"Total cost ({key})")
    results_clean.append(costs)

df_clean = pd.DataFrame(
    np.column_stack([HoursCleaning_vals] + results_clean),
    columns=["HoursCleaning"] + list(prices.keys())
)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

#plt.title("Sensitivity of HoursCleaning")
plt.tight_layout()
plt.savefig("PlotsOptimisation/HoursCleaning.png")
plt.show()

# 2. Area plot
print("Calculating Filter Area sensitivity...")
TotalEnergy_vals = []

for area in Area_vals:
    E, P = calculate_total_energy(area=area)
    TotalEnergy_vals.append(E)

plt.figure(figsize=(10, 6))
ax1 = plt.gca()

ax1.set_yscale("log")
ax1.plot(Area_vals, TotalEnergy_vals, label="Energy use", color="blue")
ax1.set_xlabel("Filter area (m²)")
ax1.set_ylabel("Total energy (kWh)", color="k")
ax1.tick_params(axis='y', labelcolor='k')
ax1.grid(True, which="both", ls="--")

ax2 = ax1.twinx()
ax2.set_yscale("log")
ax2.set_ylabel("Total cost (€)", color="k")
ax2.grid(False)

results_area = []

for key, price in prices.items():
    costs = np.array(TotalEnergy_vals) * price + Area_vals * priceM2_base
    ax2.plot(Area_vals, costs, '--', label=f"Total cost ({key})")
    results_area.append(costs)

df_area = pd.DataFrame(
    np.column_stack([Area_vals] + results_area),
    columns=["Area"] + list(prices.keys())
)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

#plt.title("Sensitivity of Filter Area")
plt.tight_layout()
plt.savefig("PlotsOptimisation/Area.png")
plt.show()

# 3d plot
# === 3D + contour plots for Area vs HoursCleaning ===
# Build grid of Area × HoursCleaning
HC_grid, A_grid = np.meshgrid(HoursCleaning_vals, Area_vals)

Energy_grid = np.zeros_like(HC_grid)
for i in range(A_grid.shape[0]):
    print(f"Calculating energy grid row {i+1} of {A_grid.shape[0]}...")
    for j in range(A_grid.shape[1]):
        Energy_grid[i, j], P = calculate_total_energy(area=A_grid[i, j],HoursCleaning=HC_grid[i, j])

# Costs
Cost_grids = {
    key: Energy_grid * price + A_grid * priceM2_base
    for key, price in prices.items()
}

# ------------------------------
# Helper: get optimum location
# ------------------------------
def find_minimum(Z_grid):
    idx = np.unravel_index(np.argmin(Z_grid), Z_grid.shape)
    A_opt = A_grid[idx]
    HC_opt = HC_grid[idx]
    Z_opt = Z_grid[idx]
    return A_opt, HC_opt, Z_opt


# ------------------------------
# 3D Surface Plot: Energy (log Z)
# ------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

logZ = np.log10(Energy_grid)
surf = ax.plot_surface(A_grid, HC_grid, logZ, cmap='plasma')

ax.set_xlabel("Filter Area (m²)")
ax.set_ylabel("Hours Between Cleanings")
ax.set_zlabel("log10(Energy kWh)")
ax.set_title("3D Surface: Energy Use (log scale)")
fig.colorbar(surf, shrink=0.5)

plt.tight_layout()
plt.savefig("PlotsOptimisation/3D_Energy_Log.png")
plt.show()


# ------------------------------
# 3D Surface Plots: Costs (log Z)
# ------------------------------
for key, Z in Cost_grids.items():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Keep z-values linear, use LogNorm for color
    surf = ax.plot_surface(A_grid, HC_grid, Z, cmap='viridis', norm=LogNorm(vmin=Z.min(), vmax=Z.max()))

    ax.set_xlabel("Filter Area (m²)")
    ax.set_ylabel("Hours Between Cleanings")
    ax.set_zlabel("Cost (€)")  # z-axis shows actual values
    ax.set_title(f"3D Surface: Total Cost ({key}) (log color scale)")

    # Colorbar with log scale
    cbar = fig.colorbar(surf, shrink=0.5)
    cbar.set_label("Cost (€)")
    cbar.locator = mticker.LogLocator(base=10.0, subs=np.linspace(1,10,10))  # nice intermediate ticks
    cbar.update_ticks()

    plt.tight_layout()
    plt.savefig(f"PlotsOptimisation/3D_Cost_{key}_LogColor.png")
    plt.show()

# Common font scaling
scale = 1
plt.rcParams.update({
    "font.size": 10 * scale,          # Base font size
    "axes.labelsize": 12 * scale,     # Axis labels
    "xtick.labelsize": 9 * scale,     # Tick labels
    "ytick.labelsize": 9 * scale,
    "legend.fontsize": 10 * scale,    # Legend
    "axes.titlesize": 12 * scale      # Titles (if enabled)
})


# ------------------------------
# Contour Plots: Energy (log scale)
# ------------------------------
special_price_points = {}
A_opt_E, HC_opt_E, E_opt = find_minimum(Energy_grid)
print("Minimum Energy:")
print("  Area =", A_opt_E)
print("  HoursCleaning =", HC_opt_E)
print("  Energy (kWh) =", E_opt)

plt.figure(figsize=(10, 7))
log_levels = np.logspace(np.log10(Energy_grid.min()),
                         np.log10(Energy_grid.max()), 40)

cp = plt.contourf(A_grid, HC_grid, Energy_grid,
                  levels=log_levels, norm="log", cmap="plasma")

# Larger optimum point
plt.scatter(A_opt_E, HC_opt_E, color="red", s=200, label="Optimum")

plt.legend()

cbar = plt.colorbar(cp)
cbar.set_label("Energy (kWh)")

# Bigger colorbar ticks
cbar.ax.tick_params(labelsize=9 * scale)

cbar.locator = mticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1)
cbar.update_ticks()

plt.xlabel("Filter Area (m²)")
plt.ylabel("Hours Between Cleanings")

plt.tight_layout()
plt.savefig("PlotsOptimisation/Contour_Energy_Log.png", dpi=300)
plt.show()


# ------------------------------
# Contour Plots: Costs (log scale)
# ------------------------------
for key, Z in Cost_grids.items():

    A_opt, HC_opt, Z_opt = find_minimum(Z)
    print(f"Minimum Cost ({key}):")
    print("  Area =", A_opt)
    print("  HoursCleaning =", HC_opt)
    print("  Cost (€) =", Z_opt)

    special_price_points[key] = {
        "Area": float(A_opt),
        "HoursCleaning": float(HC_opt),
        "Cost": float(Z_opt)
    }

    plt.figure(figsize=(10, 7))
    log_levels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 40)

    cp = plt.contourf(A_grid, HC_grid, Z,
                      levels=log_levels, norm="log", cmap='viridis')

    # Larger optimum point
    plt.scatter(A_opt, HC_opt, color="red", s=200, label="Optimum")

    plt.legend()

    cbar = plt.colorbar(cp)
    cbar.set_label("Total Cost (€)")
    cbar.ax.tick_params(labelsize=9 * scale)

    cbar.locator = mticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1)
    cbar.update_ticks()

    plt.xlabel("Filter Area (m²)")
    plt.ylabel("Hours Between Cleanings")

    plt.tight_layout()
    plt.savefig(f"PlotsOptimisation/Contour_Cost_{key}_Log.png", dpi=300)
    plt.show()


# ========== PRICE SWEEP ==========

A_list = []
HC_list = []
Z_list = []
prices2 = np.arange(0.01, 0.4, 0.000002)

for price in prices2:
    pricegrid = Energy_grid * price + A_grid * priceM2_base
    A_opt, HC_opt, Z_opt = find_minimum(pricegrid)

    A_list.append(A_opt)
    HC_list.append(HC_opt)
    Z_list.append(Z_opt)

A_list = np.array(A_list)
HC_list = np.array(HC_list)
Z_list = np.array(Z_list)


plt.figure(figsize=(10, 6))
plt.scatter(A_list, HC_list, label="Optimal pricing", color="k", s=10)

# Plot special price points (from contour minima) with labels
markers = ['o', 's', 'D', '^', 'v', 'P', 'X']
for i, (key, info) in enumerate(special_price_points.items()):
    a = info["Area"]
    hc = info["HoursCleaning"]
    plt.scatter(a, hc, marker=markers[i % len(markers)], s=150, label=f"{key} (opt)")

plt.xlabel("Optimal Filter Area (m²)")
plt.ylabel("Optimal Hours Between Cleanings (h)")
#plt.title("Optimal Filter Area vs Hours Between Cleanings over Energy Price")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("PlotsOptimisation/Optimal_Area_vs_HoursCleaning.png")
plt.show()

# ============================================================
# PLOT PRESSURE CURVES FOR EACH ENERGY PRICE OPTIMUM
# ============================================================

for key, info in special_price_points.items():
    # Use the correct optimum for this key
    area = info["Area"]
    hc   = info["HoursCleaning"]

    E, pressure = calculate_total_energy(area=area, HoursCleaning=hc)

    plt.figure(figsize=(10, 6))
    plt.plot(pressure, label=f"Optimal pressure curve ({key})")

    # Add flat dotted line at initial pressure drop
    initial_pd = pressure[0]
    plt.axhline(
        y=initial_pd,
        linestyle=':',
        linewidth=1.5,
        label="Initial pressure drop"
    )

    plt.xlabel("Time (h)")
    plt.ylabel("Transmembrane pressure drop (Pa)")
    # plt.title(f"Pressure vs Time – Optimal Conditions for {key}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"PlotsOptimisation/PressureCurve_{key}.png")
    plt.show()
    print("Plotted pressure curve for", key)
    print("max pressure (Pa):", np.max(pressure))
    print("min pressure (Pa):", np.min(pressure))

