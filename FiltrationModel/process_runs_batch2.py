# save as process_runs_batch.py and run with: python process_runs_batch.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import t
from scipy.signal import savgol_filter
from CoolProp.CoolProp import PropsSI
import FunctionFile as FF  # your musteam function

# -------------------------
# USER VARIABLES / CONSTANTS
# -------------------------
METADATA_XLSX = "Results testing2.xlsx"   # main metadata file
sheet = "Loop2"
RESULTS_XLSX = "summary_resultsloop2.xlsx" # output summary
PLOTS_DIR = "plots2"
CSV_DIR = "."                             # directory where RunXX.csv files live

# physical constants
R = 8.314
Patm = 105137
A = 0.25 * (6 * 0.01)**2 * np.pi  # membrane area (m^2)
rho_Cell = 0.2796 * 1000.0  # kg/m3
rho_CaCO3 = 0.546675 * 1000.0
ref_temp = 20 + 273.153

# ensure plot folder exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------
# load metadata
# -------------------------
meta = pd.read_excel(METADATA_XLSX, sheet_name=sheet)
meta = meta.fillna(value=np.nan)

# -------------------------
# helper functions
# -------------------------
def compute_mu_and_Qtime(Temp_array_K, Q_N2_Lmin, Q_water_mlmin):
    """Compute Q_total (m³/s) and mu arrays based on N2 and water flow"""
    Q_N2_m3s = Q_N2_Lmin / 1000.0 / 60.0
    Q_water_m3s = Q_water_mlmin / 1e6 / 60.0

    rho_water = PropsSI("D", "T", ref_temp, "P", Patm, "Water")
    mdot_water = Q_water_m3s * rho_water

    rho_N2_ref = PropsSI("D", "T", ref_temp, "P", Patm, "Nitrogen")
    mdot_N2 = Q_N2_m3s * rho_N2_ref

    Q_total_list, mu_list = [], []

    for T in Temp_array_K:
        try:
            rho_steam = PropsSI("D", "T", T, "P", Patm, "Water")
        except:
            rho_steam = 0.6  # fallback

        rho_N2_actual = PropsSI("D", "T", T, "P", Patm, "Nitrogen")
        Q_N2_actual = mdot_N2 / rho_N2_actual

        Q_steam = mdot_water / rho_steam if Q_water_m3s > 0 else 0.0
        Q_total = Q_N2_actual + Q_steam

        mu = FF.musteam(T, Q_N2_actual / Q_total if Q_total > 0 else 1.0)

        Q_total_list.append(Q_total)
        mu_list.append(mu)

    return np.array(Q_total_list), np.array(mu_list)


def fit_R_mem(P_mask, Q_mask, mu_mask, A_area):
    """Fit P = R_mem * (Q * mu / A) using direct slope estimate"""
    x = Q_mask * mu_mask / A_area
    valid = np.isfinite(P_mask) & np.isfinite(x)
    if np.sum(valid) < 2:
        return np.nan, np.nan, np.nan

    R_i = P_mask[valid] / x[valid]
    R_mean = np.mean(R_i)
    rmse = np.std(R_i, ddof=1)

    dof = len(R_i) - 1
    R_ci = t.ppf(0.975, dof) * rmse / np.sqrt(len(R_i)) if dof > 0 else np.nan
    return R_mean, rmse, R_ci


def safe_mask(time_data, start, end):
    if pd.isna(start) or pd.isna(end):
        return np.array([False] * len(time_data))
    return (time_data >= int(start)) & (time_data <= int(end))


# -------------------------
# MAIN LOOP
# -------------------------
results_rows = []
results_summary = []

for idx, row in meta.iterrows():
    try:
        file_name = str(row.get("File", "")).strip() + ".csv"
        if not os.path.exists(os.path.join(CSV_DIR, file_name)):
            print(f"Skipping row {idx}: file '{file_name}' not found.")
            continue

        print(f"\nProcessing run {row.get('Run', '')}...")

        substance = row["Substance"]
        if substance == "BWW40 Dust":
            density = rho_Cell
        elif substance == "CaCO3":
            density = rho_CaCO3
        else:
            density = None

        df = pd.read_csv(os.path.join(CSV_DIR, file_name))
        pre_p = df["Pre filter Pressure Gem. (mbar)"].fillna(0)
        post_p = df["Post filter Pressure Gem. (mbar)"].fillna(0)
        P_data = (pre_p - post_p) * 100.0  # Pa
        time_data = np.arange(1, len(P_data) + 1)
        Temp_data = df["post filter Gem. (C)"].fillna(0).to_numpy() + 273.153

        # baseline correction
        t_base_start = row.get("base_start")
        t_base_end = row.get("base_end")
        P_mask = safe_mask(time_data, t_base_start, t_base_end)
        P_data = P_data-np.mean(P_data[P_mask])  # baseline correction
        P_smooth = savgol_filter(P_data, window_length=19, polyorder=1)

        # -------------------------
        # process 10 flat zones
        # -------------------------
        zone_results = {}
        summary_entry = {
            "Run": row.get("Run", ""),
            "Substance": substance,
            "Temp": row.get("Temp", ""),
            "Flow": row.get("Flow", ""),
            "Total dust (g)": row.get("Total dust (g)", "")
        }

        plt.figure(figsize=(10, 6))
        plt.grid(True, which='major', linewidth=0.8)
        plt.minorticks_on()
        plt.gca().yaxis.set_minor_locator(MultipleLocator(100))
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5)
        plt.scatter(time_data, P_data, s=10, alpha=0.3, label="Raw data")
        plt.plot(time_data, P_smooth, label="Smoothed", linewidth=1.2)

        for i in range(1, 11):
            N2_flow = float(row.get(f"N2_{i}", 0.0) or 0.0)
            Water_flow = float(row.get(f"Water_{i}", 0.0) or 0.0)
            t_start = row.get(f"t_flat_start_{i}", np.nan)
            t_end = row.get(f"t_flat_end_{i}", np.nan)

            mask = safe_mask(time_data, t_start, t_end)
            if not mask.any():
                continue

            # compute flow and viscosity
            Q_total, mu = compute_mu_and_Qtime(Temp_data, N2_flow, Water_flow)

            #Temperature mask
            Temp_mask = Temp_data[mask]

            # fit R_mem
            R_fit, rmse, ci = fit_R_mem(P_data[mask], Q_total[mask], mu[mask], A)
            J_mean = np.mean(Q_total[mask] / A)

            # store results
            zone_results[f"R_mem_m^-1_{i}"] = R_fit
            zone_results[f"R_mem_flat_RMSE_Pa_{i}"] = rmse
            zone_results[f"R_mem_flat_CI_{i}"] = ci

            summary_entry[f"P_Pa_{i}"] = np.mean(P_data[mask])
            summary_entry[f"R_mem_m^-1_{i}"] = R_fit
            summary_entry[f"J_flat_m/s_{i}"] = J_mean
            summary_entry[f"Temp_flat_K_{i}"] = np.mean(Temp_mask)
            summary_entry[f"mu_flat_Pa.s_{i}"] = np.mean(mu[mask])
            summary_entry[f"x_m.Pa_{i}"] = J_mean*np.mean(mu[mask])

            if i>=6:
                j = 5 if i == 6 else i -6
                summary_entry[f"alpha_apparent_{i}"] = (summary_entry.get(f"P_Pa_{i}") / (
                            summary_entry.get(f"J_flat_m/s_{i}") * summary_entry.get(f"mu_flat_Pa.s_{i}"))
                                                    - summary_entry.get(f"R_mem_m^-1_{j}")) * density * A / (
                                                               row.get("Total dust (g)") / 1000.0)
                summary_entry[f"P_mem_increased_{i}"] = summary_entry.get(f"P_Pa_{i}") - summary_entry.get(f"P_Pa_{j}")
                summary_entry[f"R_mem_increased_{i}"] = summary_entry.get(f"R_mem_m^-1_{i}") - summary_entry.get(f"R_mem_m^-1_{j}")
                summary_entry[f"P_0_{i}"] = summary_entry.get(f"P_Pa_{j}")
                summary_entry[f"R_0_{i}"] = summary_entry.get(f"R_mem_m^-1_{j}")

            if i<=5:
                summary_entry[f"R_filter_m^-1_{i}"] = R_fit

            # plot fit line
            x_mask = Q_total[mask] * mu[mask] / A
            P_fit = R_fit * x_mask
            plt.plot(time_data[mask], P_fit, linewidth=2.2,
                     label=f"Flat {i} (N2={N2_flow} L/min, H2O={Water_flow} mL/min)")
        summary_entry[f"P_increase_dust"] = summary_entry.get("P_Pa_6") - summary_entry.get("P_Pa_5")
        summary_entry[f"R_mem_increase_dust"] = summary_entry.get("R_mem_m^-1_6") - summary_entry.get("R_mem_m^-1_5")


        plt.xlabel("Time (s)")
        plt.ylabel("ΔP (Pa)")
        plt.title(f"Run {row.get('Run', '')} - {file_name}")
        plt.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"run_{row.get('Run', '')}.png"), dpi=200)
        plt.close()

        # combine all results
        combined = {**{"Run": row.get("Run", ""), "File": file_name}, **zone_results}
        for c in meta.columns:
            if c not in combined:
                combined[c] = row.get(c, np.nan)

        results_rows.append(combined)
        results_summary.append(summary_entry)
        print(f"Processed run {row.get('Run', '')}")
    except Exception as e:
        print(f"Error processing row {idx}: {e}")

# -------------------------
# reorganize summary by data kind
# -------------------------
if len(results_summary) > 0:
    df_summary = pd.DataFrame(results_summary)

    # extract all possible measurement kinds
    kinds = ["P_flat", "R_mem_flat", "J_flat", "Temp_flat", "mu_flat", "alpha_apparent"]
    ordered_cols = ["Run", "Substance", "Temp", "Flow", "Total dust (g)"]

    # For each kind, collect all matching columns in natural flat order
    for kind in kinds:
        matched = sorted([c for c in df_summary.columns if c.startswith(kind)],
                         key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        ordered_cols.extend(matched)

    # include remaining metadata or derived columns
    remaining = [c for c in df_summary.columns if c not in ordered_cols]
    ordered_cols.extend(remaining)

    df_summary_sorted = df_summary[ordered_cols]
else:
    df_summary_sorted = None# -------------------------
# write to Excel
# -------------------------
if len(results_rows) > 0:
    df_results = pd.DataFrame(results_rows)
    df_summary = pd.DataFrame(results_summary)

    # (insert sorting code block here before writing)

    with pd.ExcelWriter(RESULTS_XLSX) as writer:
        df_results.to_excel(writer, sheet_name="Detailed Results", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        if df_summary_sorted is not None:
            df_summary_sorted.to_excel(writer, sheet_name="Summary_sorted_by_type", index=False)

    print(f"\nWrote results to {RESULTS_XLSX} and plots to {PLOTS_DIR}/")
else:
    print("No valid results to write.")