import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
import pandas as pd
import os
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection


# Folder for saving plots
os.makedirs("PlotsOptimisation", exist_ok=True)


# Base variable definitions
FinalWeight = 1000  # kg/h
WaterConcFinal = 0.05
WaterConcInit = 0.5
InitTempPaper = 60 + 273.15
FinalTempPaper = 150 + 273.15
InitTempSteam_base = 291 + 273.15
FinalTempSteam = 178 + 273.15
C_foulant = 1e-3  # g/m³
years = 6
FinalTime = 8600 * years  # h
priceM2_base = 80  # €/m²

# Base parameters
Area_base = 200
Area_min = 25
Area_max = 300
AreaSteps = 275

CleaningsYear_base = 2
CleaningsYear_min = 0
CleaningsYear_max = 12
CleaningsStepSize = 0.05

HoursCleaning_base = 4300
HoursCleaning_min = 100
HoursCleaning_max = 10000
HoursCleaningStepSize = 10


# Backwash Parameters
BackwashFlux = 0.1  # m/s
InletWaterTemp = 25 + 273.15
BackwashSteamTemp = 140 + 273.15
FilterFractionCleaned = 0.1
ActiveBackwashTime = 2/60     # hours
TotalBackwashTime = 1         # hours (does not affect energy)

# Scan ranges
CleaningsYear_vals = np.arange(CleaningsYear_min, CleaningsYear_max + CleaningsStepSize, CleaningsStepSize)
HoursCleaning_vals = np.arange(HoursCleaning_min, HoursCleaning_max + HoursCleaningStepSize, HoursCleaningStepSize)
Area_vals = np.linspace(Area_min, Area_max, AreaSteps)

# Energy prices €/kWh
prices = {
    "NL": 0.2046,
    "Europe": 0.1902,
    "Solar": 0.037,
    "Wind": 0.029
    }

# Total energy calculation
def calculate_total_energy(area=Area_base, HoursCleaning=HoursCleaning_base, init_steam=InitTempSteam_base):
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

    for i, t in enumerate(time):
        time_since_clean = (t % hPerClean)
        # Run backwash once per cycle

        if t >= hPerClean and (t % hPerClean) < 1:
            IrDust = 0.52 * C_foulant * volumetric_flow * 3600

            # Active backwash steam energy
            energy_active_backwash = (rho_steam_backwash * BackwashFlux * area *
                                     (h_steam_backwash - h_water_inlet_backwash)) * ActiveBackwashTime #Wh

            energy_active_backwash /= 1e3  # kWh
            TotalEnergyActiveBackwash += energy_active_backwash
            TotalEnergy_kWh += energy_active_backwash
            #print("hi")

        # Fouling pressure + motor energy
        M_t = C_foulant * volumetric_flow * time_since_clean * 3600 + IrDust
        if t >= hPerClean and (t % hPerClean) < 1:
            J = volumetric_flow / (area * (1 - FilterFractionCleaned))
        else:
            J = volumetric_flow / area


        P0 = (1.464e8 + (J * mu_steam)**2 * FinalTempSteam) * J * mu_steam
        Pt = P0 + 8.188e8 * (J * mu_steam)**2 * FinalTempSteam * M_t / area

        E_kW = Pt * volumetric_flow / 1000

        if i > 0:
            dt = time[i] - time[i - 1]
            TotalEnergy_kWh += 0.5 * (E_kW + prev_E_kW) * dt

        prev_E_kW = E_kW
    return TotalEnergy_kWh

# 1. HoursCleaning plot
TotalEnergy_vals = []
HoursCleaning_vals = np.array([h for h in HoursCleaning_vals if FinalTime % h == 0])
print(HoursCleaning_vals)
for cy in HoursCleaning_vals:
    E = calculate_total_energy(HoursCleaning=cy)
    TotalEnergy_vals.append(E)

plt.figure(figsize=(10, 6))
ax1 = plt.gca()

ax1.plot(HoursCleaning_vals, TotalEnergy_vals, label="Energy use (kWh)", color="blue")
ax1.set_yscale("log")
ax1.set_xlabel("Hours in between cleaning (h)")
ax1.set_ylabel("Total energy (kWh)", color="blue")
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, which="both", ls="--")

ax2 = ax1.twinx()
ax2.set_yscale("log")
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

plt.title("Sensitivity of HoursCleaning")
plt.tight_layout()
plt.savefig("PlotsOptimisation/HoursCleaning.png")
plt.show()

# 2. Area plot
TotalEnergy_vals = []

for area in Area_vals:
    E = calculate_total_energy(area=area)
    TotalEnergy_vals.append(E)

plt.figure(figsize=(10, 6))
ax1 = plt.gca()

ax1.set_yscale("log")
ax1.plot(Area_vals, TotalEnergy_vals, label="Energy use (kWh)", color="blue")
ax1.set_xlabel("Filter area (m²)")
ax1.set_ylabel("Total energy (kWh)", color="blue")
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, which="both", ls="--")

ax2 = ax1.twinx()
ax2.set_yscale("log")
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

plt.title("Sensitivity of Filter Area")
plt.tight_layout()
plt.savefig("PlotsOptimisation/Area.png")
plt.show()

# 3d plot
# === 3D + contour plots for Area vs HoursCleaning ===
# Build grid of Area × HoursCleaning
HC_grid, A_grid = np.meshgrid(HoursCleaning_vals, Area_vals)

Energy_grid = np.zeros_like(HC_grid)
for i in range(A_grid.shape[0]):
    for j in range(A_grid.shape[1]):
        Energy_grid[i, j] = calculate_total_energy(area=A_grid[i, j],
                                                   HoursCleaning=HC_grid[i, j])

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
surf = ax.plot_surface(A_grid, HC_grid, logZ, cmap='viridis')

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

    surf = ax.plot_surface(A_grid, HC_grid, np.log10(Z), cmap='plasma')

    ax.set_xlabel("Filter Area (m²)")
    ax.set_ylabel("Hours Between Cleanings")
    ax.set_zlabel("log10(Cost €)")
    ax.set_title(f"3D Surface: Total Cost ({key}) (log scale)")
    fig.colorbar(surf, shrink=0.5)

    plt.tight_layout()
    plt.savefig(f"PlotsOptimisation/3D_Cost_{key}_Log.png")
    plt.show()


# ------------------------------
# Contour Plots: Energy (log scale)
# ------------------------------
A_opt_E, HC_opt_E, E_opt = find_minimum(Energy_grid)
print("Minimum Energy:")
print("  Area =", A_opt_E)
print("  HoursCleaning =", HC_opt_E)
print("  Energy (kWh) =", E_opt)

plt.figure(figsize=(10, 7))
log_levels = np.logspace(np.log10(Energy_grid.min()),
                         np.log10(Energy_grid.max()), 40)

cp = plt.contourf(A_grid, HC_grid, Energy_grid,
                  levels=log_levels, norm="log", cmap="viridis")

plt.scatter(A_opt_E, HC_opt_E, color="red", s=50, label="Optimum")
plt.legend()

plt.colorbar(cp)
plt.xlabel("Filter Area (m²)")
plt.ylabel("Hours Between Cleanings")
plt.title("Contour: Energy Use (log scale)")
plt.tight_layout()
plt.savefig("PlotsOptimisation/Contour_Energy_Log.png")
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

    plt.figure(figsize=(10, 7))
    log_levels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 40)

    cp = plt.contourf(A_grid, HC_grid, Z,
                      levels=log_levels, norm="log", cmap="plasma")

    plt.scatter(A_opt, HC_opt, color="red", s=50, label="Optimum")
    plt.legend()

    plt.colorbar(cp)
    plt.xlabel("Filter Area (m²)")
    plt.ylabel("Hours Between Cleanings")
    plt.title(f"Contour: Total Cost ({key}) (log scale)")
    plt.tight_layout()
    plt.savefig(f"PlotsOptimisation/Contour_Cost_{key}_Log.png")
    plt.show()

#4 plot with all energy prices
# Arrays to store results
A_list = []
HC_list = []
price_list = []

pricearray = np.arange(0.01, 0.4, 0.001)

for price in pricearray:
    pricegrid = Energy_grid * price + A_grid * priceM2_base
    A_opt, HC_opt, Z_opt = find_minimum(pricegrid)

    # Store results
    A_list.append(A_opt)
    HC_list.append(HC_opt)
    price_list.append(price)

df_opt = pd.DataFrame({
    "Price [€/kWh]": price_list,
    "A_opt [m²]": A_list,
    "HC_opt [h]": HC_list
})
# Prepare points
points = np.array([A_list, HC_list]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a LineCollection with color based on price
lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(min(price_list), max(price_list)))
lc.set_array(np.array(price_list))
lc.set_linewidth(2)

# Plot
fig, ax = plt.subplots(figsize=(8,6))
ax.add_collection(lc)
ax.set_xlim(min(A_list), max(A_list))
ax.set_ylim(min(HC_list), max(HC_list))
ax.set_xlabel("Optimal Area [m²]")
ax.set_ylabel("Optimal Cleaning Interval [h]")
ax.set_title("Optimal (A, HC) Across Electricity Prices")
ax.grid(True)

# Add colorbar
cbar = fig.colorbar(lc, ax=ax)
cbar.set_label("Electricity Price [€/kWh]")

plt.tight_layout()
plt.show()