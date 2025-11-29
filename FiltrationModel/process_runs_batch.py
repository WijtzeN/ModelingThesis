# save as process_runs_batch.py and run with: python process_runs_batch.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from scipy.stats import t
from scipy.signal import savgol_filter
from CoolProp.CoolProp import PropsSI
import FunctionFile as FF  # your musteam function

# -------------------------
# USER VARIABLES / CONSTANTS
# -------------------------
METADATA_XLSX = "Results testing2.xlsx"     # main metadata file
sheet = "Loop"
RESULTS_XLSX = "summary_resultsloop.xlsx"    # output summary
PLOTS_DIR = "plots"
CSV_DIR = "."                            # directory where RunXX.csv files live (change if needed)

Q_N2_Lmin = 3.55  # fixed N2 flow in L/min
Q_N2_m3s_ref = Q_N2_Lmin / 1000.0 / 60.0 #m3/s

# physical constants (from your original code)
R = 8.314
Patm = 105137
A = 0.25 * (6 * 0.01)**2 * np.pi  # membrane area (m^2)
rho_Cell = 0.2796 * 1000.0  # kg/m3 (your previous value)
rho_CaCO3 = 0.546675 * 1000.0
ref_temp = 20 + 273.153


# smoothing params
SMOOTH_WINDOW = 9
SMOOTH_POLYORDER = 1

# ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------
# load metadata table
# -------------------------
meta = pd.read_excel(METADATA_XLSX, sheet_name=sheet)
meta = meta.fillna(value=np.nan)  # keep NaNs explicit

# -------------------------
# helper functions
# -------------------------
def compute_mu_and_Qtime(Temp_array_K, Q_water_mlmin):
    """
    Compute arrays for Q_total (m^3/s), Q_N2_only (m^3/s) and mu_time (Pa s)
    given a Temp array (K) and water flow in mL/min. Uses FF.musteam(T, steam_fraction)
    """
    Q_water_m3s = Q_water_mlmin / 1000.0 / 1000.0 / 60.0  # m^3/s
    rho_water = PropsSI("D", "T", ref_temp, "P", Patm, "Water")  # density of liquid water at ref temp
    mdot_water = Q_water_m3s * rho_water

    # reference N2 density at ref temp
    rho_N2_ref = PropsSI("D", "T", ref_temp, "P", Patm, "Nitrogen")
    mdot_N2 = Q_N2_m3s_ref * rho_N2_ref  # mass flow of N2 (kg/s) - constant

    Q_total_list = []
    Q_N2_only_list = []
    mu_list = []

    for T in Temp_array_K:
        # water/steam density at T
        # use CoolProp density for water vapor at local T and (Patm) pressure
        try:
            rho_steam = PropsSI("D", "T", T, "P", Patm, "Water")
        except Exception as e:
            # fallback to small number to avoid zero division if Props fails
            print("PropsSI failed for steam density at T=", T)
            print(e)


        # steam volumetric flow
        if Q_water_m3s == 0.0:
            Q_steam_only = 0.0
        else:
            # mdot_water / rho_steam gives volumetric steam flow m^3/s
            Q_steam_only = mdot_water / rho_steam

        # N2 density at T
        rho_N2_actual = PropsSI("D", "T", T, "P", Patm, "Nitrogen")
        Q_N2_m3s_actual = mdot_N2 / rho_N2_actual

        Q_total = Q_steam_only + Q_N2_m3s_actual
        Q_total_list.append(Q_total)
        Q_N2_only_list.append(Q_N2_m3s_actual)
        mu = FF.musteam(T, Q_N2_m3s_actual / Q_total)
        mu_list.append(mu)

    return np.array(Q_total_list), np.array(Q_N2_only_list), np.array(mu_list)


def fit_R_mem(P_mask, Q_mask, mu_mask, A_area):
    """ Fit P = R_mem * (Q * mu / A) using curve_fit, return R_mem, rmse, R_mem_ci """
    # calculate x = Q*mu/A
    x = Q_mask * mu_mask / A_area

    # valid points only
    mask_valid = np.isfinite(x) & np.isfinite(P_mask)
    if np.sum(mask_valid) < 2:
        return np.nan, np.nan, np.nan

    P_valid = P_mask[mask_valid]
    x_valid = x[mask_valid]

    # instantaneous R_i = P / (Q*mu/A)
    R_i = P_valid / x_valid

    # mean R_mem
    R_mem = np.mean(R_i)

    # standard deviation (for RMSE)
    rmse = np.std(R_i, ddof=1)

    # 95% CI using t-distribution
    dof = len(R_i) - 1
    if dof > 0:
        tval = t.ppf(0.975, dof)
        R_mem_ci = tval * rmse / np.sqrt(len(R_i))
    else:
        R_mem_ci = np.nan

    return R_mem, rmse, R_mem_ci

def safe_mask(time_data, start, end):
    if start is None or end is None:
        return np.array([False] * len(time_data))
    return (time_data >= start) & (time_data <= end)


results_rows = []
results_summery = []

# loop runs
for idx, row in meta.iterrows():
    try:
        file_name = str(row.get("File", "")).strip() + ".csv"
        if file_name == "" or not os.path.exists(os.path.join(CSV_DIR, file_name)):
            print(f"Skipping row {idx}: file '{file_name}' not found.")
            continue
        substance = row["Substance"]
        if substance == "BWW40 Dust":
            density = rho_Cell  # g/m3
        elif substance == "CaCO3":
            density = rho_CaCO3  # g/m3
        else:
            density = None

        # read CSV
        df = pd.read_csv(os.path.join(CSV_DIR, file_name))

        # pressures and temperature
        pre_p = df["Pre filter Pressure Gem. (mbar)"].fillna(0)
        post_p = df["Post filter Pressure Gem. (mbar)"].fillna(0)
        P_data = (pre_p - post_p) * 100.0  # mbar -> Pa
        P_data = P_data.reset_index(drop=True).to_numpy()
        time_data = np.arange(1, len(P_data) + 1)
        Temp_data = df["post filter Gem. (C)"].fillna(0).to_numpy() + 273.153

        # baseline correction (if provided via metadata 'base_bounds' use it, else default 0..1)
        base0 = int(row.get("base_start", 0)) if not np.isnan(row.get("base_start", np.nan)) else 0
        base1 = int(row.get("base_end", 1)) if not np.isnan(row.get("base_end", np.nan)) else 1
        correction_P_data = np.mean(P_data[base0:base1]) if (base1 > base0) else 0.0
        P_data = P_data - correction_P_data

        # smooth
        try:
            #P_data_smooth = P_data.ewm(alpha=0.05).mean()
            s = pd.Series(P_data)
            smoothed = s.ewm(alpha=0.05).mean()
            smoothed_array = smoothed.values
            P_data_smooth = smoothed.to_numpy()
            #P_data_smooth = savgol_filter(P_data, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER)
        except Exception as e:
            # in case series too short
            print(e)
            P_data_smooth = P_data.copy()
        # Run-specific constants
        try:
            M_g = row["Total dust (g)"]
        except (ValueError, TypeError):
            print(f"Error converting mass in row {idx}: {M_g}")

        # extract time boundaries from metadata (expect integers)
        def get_int(col):
            v = row.get(col, np.nan)
            return int(v) if not (pd.isna(v)) else None

        f1s = get_int("t_flat1_start"); f1e = get_int("t_flat1_end")
        f2s = get_int("t_flat2_start"); f2e = get_int("t_flat2_end")
        f3s = get_int("t_flat3_start"); f3e = get_int("t_flat3_end")
        f4s = get_int("t_flat4_start"); f4e = get_int("t_flat4_end")
        rs = get_int("t_rise_start"); re = get_int("t_rise_end")

        # if None in (f1s, f1e, f2s, f2e, f3s, f3e, f4s, f4e, rs, re):
        #     print(f"Row {idx} missing some time bounds — skipping.")
        #     continue

        # Q water values from metadata (ml/min). If NaN default to 0 for flats 1/3
        Q_f2 = float(row.get("Q_flat2_mlmin", 0.0)) if not pd.isna(row.get("Q_flat2_mlmin", np.nan)) else 0.0
        Q_rise = float(row.get("Q_rise_mlmin", 0.0)) if not pd.isna(row.get("Q_rise_mlmin", np.nan)) else 0.0
        Q_backwash = float(row.get("Q_backwash_mlmin", 0.0)) if not pd.isna(row.get("Q_backwash_mlmin", np.nan)) else 0.0
        Q_f4 = float(row.get("Q_flat4_mlmin", 0.0)) if not pd.isna(row.get("Q_flat4_mlmin", np.nan)) else 0.0
        print(f"Processing run {row.get('Run', '')} with Q_f2={Q_f2}, Q_rise={Q_rise}, Q_f4={Q_f4}")

        # compute flow+mu arrays for each Q condition we need:
        # For flats 1 and 3: Q_water = 0
        Q_total_0, Q_N2_only_0, mu_0 = compute_mu_and_Qtime(Temp_data, Q_water_mlmin=0.0)
        # For flat2:
        Q_total_2, Q_N2_only_2, mu_2 = compute_mu_and_Qtime(Temp_data, Q_water_mlmin=Q_f2)
        # For rise:
        Q_total_r, Q_N2_only_r, mu_r = compute_mu_and_Qtime(Temp_data, Q_water_mlmin=Q_rise)
        # For backwash
        Q_total_bw, Q_N2_only_bw, mu_bw = compute_mu_and_Qtime(Temp_data, Q_water_mlmin=Q_backwash)
        # For flat4:
        Q_total_4, Q_N2_only_4, mu_4 = compute_mu_and_Qtime(Temp_data, Q_water_mlmin=Q_f4)

        # masks (note: time_data is 1-indexed as in your code)
        mask_f1 = safe_mask(time_data, f1s, f1e)
        mask_f2 = safe_mask(time_data, f2s, f2e)
        mask_f3 = safe_mask(time_data, f3s, f3e)
        mask_f4 = safe_mask(time_data, f4s, f4e)
        mask_rise = safe_mask(time_data, rs, re)

        # P masks
        P_f1 = P_data[mask_f1]; P_f2 = P_data[mask_f2]; P_f3 = P_data[mask_f3]; P_f4 = P_data[mask_f4]
        P_r = P_data[mask_rise]

        # Q and mu masks (use appropriate arrays)
        Q_f1 = Q_N2_only_0[mask_f1]  # N2 only
        mu_f1 = mu_0[mask_f1]

        Q_f2 = Q_total_2[mask_f2]    # N2+steam
        mu_f2 = mu_2[mask_f2]

        Q_f3 = Q_N2_only_0[mask_f3]  # after cleaning N2-only
        mu_f3 = mu_0[mask_f3]

        Q_f4 = Q_total_4[mask_f4]    # after cleaning N2+steam
        mu_f4 = mu_4[mask_f4]

        # Fit R_mem for each flat
        R_f1, rmse_f1, ci_f1 = fit_R_mem(P_f1, Q_f1, mu_f1, A)
        R_f2, rmse_f2, ci_f2 = fit_R_mem(P_f2, Q_f2, mu_f2, A)
        R_f3, rmse_f3, ci_f3 = fit_R_mem(P_f3, Q_f3, mu_f3, A)
        R_f4, rmse_f4, ci_f4 = fit_R_mem(P_f4, Q_f4, mu_f4, A)

        # --- averages for flats ---
        P_f1, J_f1, T_f1 = (np.mean(P_data[mask_f1]), np.mean(Q_N2_only_0[mask_f1] / A), np.mean(Temp_data[mask_f1])) if mask_f1.any() else (np.nan, np.nan)
        P_f2, J_f2, T_f2 = (np.mean(P_data[mask_f2]), np.mean(Q_total_2[mask_f2] / A), np.mean(Temp_data[mask_f2])) if mask_f2.any() else (np.nan, np.nan)
        P_f3, J_f3, T_f3 = (np.mean(P_data[mask_f3]), np.mean(Q_N2_only_0[mask_f3] / A), np.mean(Temp_data[mask_f3])) if mask_f3.any() else (np.nan, np.nan)
        P_f4, J_f4, T_f4 = (np.mean(P_data[mask_f4]), np.mean(Q_total_4[mask_f4] / A), np.mean(Temp_data[mask_f4])) if mask_f4.any() else (np.nan, np.nan)


        # Peak & alpha using smoothed P in rise mask
        if np.sum(mask_rise) > 0:
            P_smooth_rise = P_data_smooth[mask_rise]
            max_P_smooth_rise = np.max(P_smooth_rise)
            # P_min from flat fits: use minimum of fitted flat P (we can approximate P_min from flat1 fit)
            # Build P_fit_flat for whichever flat is baseline: use flat1 fit to derive P_min
            # Compute P_min estimate = R_f1 * mean(x_flat1)
            x_flat1 = (Q_N2_only_0[mask_f1] * mu_0[mask_f1]) / A if np.sum(mask_f1) > 0 else np.array([0.0])
            mean_x_flat1 = np.mean(x_flat1) if x_flat1.size > 0 else 0.0
            #P_min_est = R_f1 * mean_x_flat1 if np.isfinite(R_f1) else np.min(P_data_smooth)

            # find the global idx of the peak (map local mask index to full time index)
            idx_local_peak = np.argmax(P_data_smooth[mask_rise])
            idx_global_peak = np.where(mask_rise)[0][0] + idx_local_peak

            # Q_peak and mu_peak need the Q/mu arrays corresponding to the chosen Q for rise
            Q_peak = Q_total_r[idx_global_peak]
            mu_peak = mu_r[idx_global_peak]
            # R_peak as earlier:
            R_peak = max_P_smooth_rise * A / (Q_peak * mu_peak) if (Q_peak * mu_peak) != 0 else np.nan

            P_increase = max_P_smooth_rise - P_f2#P_min_est
            # alpha formula same as your code: (max_deltaP * rho_Cell * A^2) / (Q_peak * mu_peak * M_kg)
            M_kg = M_g / 1000.0
            alpha_val = (P_increase * density * A**2) / (Q_peak * mu_peak * M_kg) if (Q_peak * mu_peak * M_kg) != 0 else np.nan
            print(f"Run {row.get('Run', '')}: alpha={alpha_val:.3e}, P_increase={P_increase:.1f} Pa, density={density}, M_kg={M_kg}, Q_peak={Q_peak}, mu_peak={mu_peak}, A={A}")
            J_rise = np.mean(Q_total_r[mask_rise] / A)
            T_rise = np.mean(Temp_data[mask_rise])
        else:
            R_peak = np.nan
            alpha_val = np.nan
            P_increase = np.nan

        # compile results row
        result = {
            "Run": row.get("Run", ""),
            "File": file_name,
            "M_g": M_g,
            "Q_N2_Lmin": Q_N2_Lmin,

            # Resistances
            "R_mem_flat1_N2_m^-1": R_f1,
            "R_mem_flat1_RMSE_Pa": rmse_f1,
            "R_mem_flat1_CI": ci_f1,

            "R_mem_flat2_N2plusSteam_m^-1": R_f2,
            "R_mem_flat2_RMSE_Pa": rmse_f2,
            "R_mem_flat2_CI": ci_f2,

            "R_mem_flat3_N2_cleaned_m^-1": R_f3,
            "R_mem_flat3_RMSE_Pa": rmse_f3,
            "R_mem_flat3_CI": ci_f3,

            "R_mem_flat4_N2plusSteam_cleaned_m^-1": R_f4,
            "R_mem_flat4_RMSE_Pa": rmse_f4,
            "R_mem_flat4_CI": ci_f4,

            "R_peak_m^-1": R_peak,
            "alpha_m^-2": alpha_val,
            "P_increase_rise_Pa": P_increase
        }

        results_summery_row = {
            "Run": row.get("Run", ""),
            "Filter": row.get("Filter", ""),
            "Substance": substance,
            "Temp": row.get("Temp",""),
            "Flow": row.get("Flow",""),
            "Backwash_Flow": row.get("Backwash Flow",""),
            "Clean_time": row.get("Clean time",""),
            "Total dust (g)": M_g,
            "Dust in filter (g)": row.get("Dust in filter (g)",""),
            "R_mem_flat1_N2_m^-1": R_f1,
            "R_mem_flat2_N2plusSteam_m^-1": R_f2,
            "R_mem_flat3_N2_cleaned_m^-1": R_f3,
            "R_mem_flat4_N2plusSteam_cleaned_m^-1": R_f4,
            "R_peak_m^-1": R_peak,
            "alpha_m^-2": alpha_val,
            "P_flat1_Pa": P_f1,
            "J_flat1_m/s": J_f1,
            "mu_flat1_Pa.s": np.mean(mu_0[mask_f1]),
            "T_flat1_K": T_f1,
            "P_flat2_Pa": P_f2,
            "J_flat2_m/s": J_f2,
            "mu_flat2_Pa.s": np.mean(mu_2[mask_f2]),
            "T_flat2_K": T_f2,
            "P_flat3_Pa": P_f3,
            "J_flat3_m/s": J_f3,
            "mu_flat3_Pa.s": np.mean(mu_0[mask_f3]),
            "T_flat3_K": T_f3,
            "P_flat4_Pa": P_f4,
            "J_flat4_m/s": J_f4,
            "mu_flat4_Pa.s": np.mean(mu_4[mask_f4]),
            "T_flat4_K": T_f4,
            "P_increase_rise_Pa": P_increase,
            "P_peak": max_P_smooth_rise,
            "J_rise_m/s": J_rise,
            "mu_rise_Pa.s": np.mean(mu_r[mask_rise]),
            "T_rise_K": T_rise,
            'New resistance N2': R_f3 - R_f1,
            'New resistance N2+Steam': R_f4 - R_f2,

        }

        # copy other metadata fields to output for reference
        for c in meta.columns:
            if c not in result:
                result[f"{c}"] = row.get(c, np.nan)
                results_summery_row[f"{c}"] = row.get(c, np.nan)

        results_rows.append(result)
        results_summery.append(results_summery_row)

        # -------------
        # plotting
        # -------------
        plt.figure(figsize=(10, 6))
        plt.grid(True, which='major', linewidth=0.8)
        plt.minorticks_on()
        plt.gca().yaxis.set_minor_locator(MultipleLocator(100))
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5)

        tt = time_data
        plt.scatter(tt, P_data, label='Raw data', alpha=0.3, s=10)
        plt.plot(tt, P_data_smooth, label='Smoothed', linewidth=1)


        # Overlay linear fit lines on corresponding flat time intervals (reconstruct x and plot)
        def plot_fit_on_mask(mask, R_val, Q_arr, mu_arr, label_text):
            if np.isnan(R_val) or np.sum(mask) == 0:
                return
            t_mask = time_data[mask]
            x_mask = (Q_arr[mask] * mu_arr[mask]) / A
            P_fit_mask = R_val * x_mask

            ### Calculate flux J = Q/A for this mask
            J_mask = Q_arr[mask] / A
            avg_J = np.nanmean(J_mask) if len(J_mask) > 0 else np.nan

            ### Update label to include flux in legend
            label_full = f"{label_text} (J={avg_J:.3e} m/s)"

            plt.plot(t_mask, P_fit_mask, linewidth=2.5, label=label_full)


        plot_fit_on_mask(mask_f1, R_f1, Q_N2_only_0, mu_0, "Fit flat1 (N2)")
        plot_fit_on_mask(mask_f2, R_f2, Q_total_2, mu_2, "Fit flat2 (N2+steam)")
        plot_fit_on_mask(mask_f3, R_f3, Q_N2_only_0, mu_0, "Fit flat3 (N2 cleaned)")
        plot_fit_on_mask(mask_f4, R_f4, Q_total_4, mu_4, "Fit flat4 (N2+steam cleaned)")

        plt.xlabel("Time (s)")
        plt.ylabel("ΔP (Pa)")
        plt.title(f"Run {result['Run']} - {file_name}")
        plt.legend(loc='best', fontsize=9)
        plt.xlim([0, max(time_data)])
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"run_{result['Run']}_{os.path.splitext(file_name)[0]}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"Processed run {result['Run']} -> R_f1={R_f1:.2e}, R_f2={R_f2:.2e}, R_peak={R_peak:.2e}")

    except Exception as e:
        print(f"Error processing row {idx}: {e}")

# -------------------------
# write results to excel
# -------------------------
if len(results_rows) > 0:
    df_results = pd.DataFrame(results_rows)
    df_summery = pd.DataFrame(results_summery)
    # Save detailed results and summary to separate sheets
    with pd.ExcelWriter(RESULTS_XLSX) as writer:
        df_results.to_excel(writer, sheet_name="Detailed Results", index=False)
        df_summery.to_excel(writer, sheet_name="Summary", index=False)
    print(f"Wrote summary to {RESULTS_XLSX} and plots to {PLOTS_DIR}/")
else:
    print("No results to write.")
