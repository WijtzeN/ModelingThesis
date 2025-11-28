import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
from tabulate import tabulate
import itertools
import os
import matplotlib.pyplot as plt

# --- 1. Load Data ---
file = "summary_resultsloop2.xlsx"
df = pd.read_excel(file, sheet_name="Summary", nrows=44)
write_file = 'Ancova_Loop2_2.xlsx'
plot_folder = "ANCOVA_Plots"
# --- Ensure plot folder exists ---
os.makedirs(plot_folder, exist_ok=True)
# --- Helper: Filter Data ---
def filter_dataframe(df, filters=None):
    if filters:
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in filters.items():
            if isinstance(val, (list, tuple, set)):
                mask &= df[col].isin(val)
            else:
                mask &= df[col] == val
        return df.loc[mask]
    else:
        return df

# Apply filters
my_filters = {'Substance': 'BWW40 Dust'}
df = filter_dataframe(df, my_filters)

# --- Variables ---
controlled_factors = ['C(Substance)', 'C(Temp)', 'C(Flow)']
semi_controlled_covariates = [
    'Total dust (g)',
    "J_flat_m/s",
    "mu_flat_Pa.s",
    "P_0",
]
# Extract the real column names behind C()
cat_cols = [c.replace("C(", "").replace(")", "") for c in controlled_factors]
all_variables = controlled_factors + semi_controlled_covariates
controlled_names = [cf[2:-1] if cf.startswith('C(') and cf.endswith(')') else cf for cf in controlled_factors]

# Remove variables already in filters or controlled
filters_keys_extended = list(my_filters.keys()) + [f'C({key})' for key in my_filters.keys()]
combined_filter_keys = set(filters_keys_extended + controlled_names)
all_variables = [var for var in all_variables if var not in combined_filter_keys and f'C({var})' not in combined_filter_keys]

print("Variables considered for ANCOVA:", all_variables)

# --- Prepare long-format for replicates ---
replicate_vars = [
    "P_Pa", "R_mem_m^-1", "J_flat_m/s", "Temp_flat_K", "mu_flat_Pa.s",
    "x_m.Pa", "alpha_apparent", "P_0", "R_0", "R_filter_m^-1"
]

long_rows = []
for _, row in df.iterrows():
    run = row["Run"]
    for i in range(1, 11):
        new_row = row.copy()
        new_row["Run_id"] = f"{run}_{i}"
        for var in replicate_vars:
            col_name = f"{var}_{i}"
            new_row[var] = row[col_name] if col_name in df.columns else np.nan
        if "Total dust (g)" in new_row and i <= 5:
            new_row["Total dust (g)"] = 0
        long_rows.append(new_row)

df_long = pd.DataFrame(long_rows)
df = df_long

list_of_results = ["P_Pa", "R_mem_m^-1", "R_filter_m^-1", "alpha_apparent"]

# --- Formula helper ---
def build_formula_ancova(result, variables):
    formula_parts = []
    for var in variables:
        if var in controlled_factors:
            formula_parts.append(var)
        else:
            formula_parts.append(f'Q("{var}")')
    return f'Q("{result}") ~ ' + ' + '.join(formula_parts)

# --- Run models ---
screening_results = []
all_model_params = []

# Generate combinations: 1 to 9 variables
combinations_to_test = []
for r in range(1, 10):
    combinations_to_test.extend(itertools.combinations(all_variables, r))


for result in list_of_results:
    print(f"Testing result: {result}")
    for variable_comb in combinations_to_test:
        try:
            formula = build_formula_ancova(result, variable_comb)
            model = ols(formula, data=df).fit()
            screening_results.append({
                'result': result,
                'variables': ', '.join(variable_comb),
                'n_variables': len(variable_comb),
                'rsquared': model.rsquared,
                'adj_rsquared': model.rsquared_adj,
                'p_value': model.f_pvalue,
                'aic': model.aic,
                'bic': model.bic
            })
            # Save parameter details
            for param, coef in model.params.items():
                all_model_params.append({
                    "Result": result,
                    "Variables": ", ".join(variable_comb),
                    "Param": param,
                    "Coef": coef,
                    "StdErr": model.bse[param],
                    "t": model.tvalues[param],
                    "p_value": model.pvalues[param],
                    "Adj R²": model.rsquared_adj,
                    "AIC": model.aic,
                    "BIC": model.bic,
                })
        except Exception as e:
            continue

results_df = pd.DataFrame(screening_results)
all_model_params_df = pd.DataFrame(all_model_params)

# --- Identify best models ---
results_df['f_stat_significant'] = results_df['p_value'] < 0.05
best_models_bic_short = results_df.sort_values(['result', 'bic']).groupby('result').head(1)

# --- ANCOVA tables using the longest model from previous loop ---
ancova_results = []

for result in list_of_results:
    # Find the longest variable combination for this result from screening results
    subset = results_df[results_df['result'] == result]
    if subset.empty:
        print(f"No models found for {result}")
        continue

    # Get the combination with the most variables
    longest_row = subset.sort_values('n_variables', ascending=False).iloc[0]
    variables_in_model = [v.strip() for v in longest_row['variables'].split(',')]


    print(f"Testing result: {result} with variables: {variables_in_model}")
    if result == "R_filter_m^-1":
        variables_in_model = [v for v in variables_in_model if v != "Total dust (g)"]
    # Build formula and fit
    formula = build_formula_ancova(result, variables_in_model)
    try:
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)

        for var in anova_table.index:
            row = anova_table.loc[var]
            ancova_results.append({
                'result': result,
                'variable': var,
                'sum_sq': row['sum_sq'],
                'df': row['df'],
                'F': row['F'],
                'PR(>F)': row['PR(>F)']
            })
    except Exception as e:
        print(f"Error in model for {result}: {e}")

full_ancova_df = pd.DataFrame(ancova_results)


# --- Print summary tables ---
for result in list_of_results:
    print(f"\nFull ANCOVA summary for {result}")
    res_df = full_ancova_df[full_ancova_df['result'] == result].sort_values('F', ascending=False)
    print(tabulate(res_df[['variable', 'F', 'PR(>F)']], headers='keys', tablefmt='github', showindex=False))

# --- Save everything to Excel ---
with pd.ExcelWriter(write_file, engine='openpyxl') as writer:
    df_long.to_excel(writer, sheet_name='Data_Tested', index=False)
    results_df.to_excel(writer, sheet_name='All_Model_Tests', index=False)
    all_model_params_df.to_excel(writer, sheet_name='All_Model_Params', index=False)
    full_ancova_df.to_excel(writer, sheet_name='Full_ANCOVA', index=False)

print("ANCOVA analysis complete. All results saved to", write_file)

# --- Summary of best models by BIC ---
best_models_summary = results_df.sort_values(['result', 'bic']).groupby('result').head(1)
ordered_cols = ['result', 'variables', 'rsquared', 'adj_rsquared', 'aic', 'bic',
                'n_variables', 'p_value', 'f_stat_significant']
best_models_summary = best_models_summary[ordered_cols]
print(tabulate(best_models_summary, headers='keys', tablefmt='github', showindex=False))

# Settings
sort_by = 'bic'  # Can also use 'rsquared', 'p_value', etc.
top_n = 5

for result in list_of_results:

    # Top N logic
    top_n_act = 2*top_n if result == "R_filter_m^-1" else top_n

    # Filter models and sort
    subset = (
        results_df[results_df['result'] == result]
        .sort_values(sort_by, ascending=True)
        .head(top_n_act)
        .copy()
    )

    # Remove "Total dust (g)" models only for R_filter_m^-1
    if result == "R_filter_m^-1":
        subset = subset[~subset['variables'].str.contains(r"Total dust \(g\)", regex=True, na=False)]

    if subset.empty:
        continue

    # Clean p-values
    subset['p_value'] = pd.to_numeric(subset['p_value'], errors='coerce')
    subset = subset.dropna(subset=['p_value'])
    subset = subset[subset['p_value'] > 0]   # remove 0 or negative (avoids log error)

    if subset.empty:
        print(f"No valid models to plot for {result}")
        continue

    # Prepare plot data
    pvals = subset['p_value'].to_numpy()
    x = -np.log10(pvals)
    y = subset['variables'].to_numpy()
    colors = ['skyblue' if p < 0.05 else 'red' for p in pvals]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.barh(y, x, color=colors)
    plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')

    plt.xlabel('-log10(p-value)')
    plt.title(f'Top {top_n} models for {result} sorted by {sort_by}')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plot_folder, f'{result}_top_models.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot for {result} to {plot_path}")

sort_by = 'bic'  # your criterion for best model
top_n = 1  # only the best model

for result in list_of_results:
    # Select best model
    subset_models = results_df[results_df['result'] == result].sort_values(sort_by,
                                                                           ascending=(sort_by in ['aic', 'bic']))
    if subset_models.empty:
        continue
    best_model = subset_models.iloc[0]
    variables_in_model = [v.strip() for v in best_model['variables'].split(',')]

    # Remove Total dust for R_filter if needed
    if result == "R_filter_m^-1":
        variables_in_model = [v for v in variables_in_model if v != "Total dust (g)"]

    # Fit model
    formula = build_formula_ancova(result, variables_in_model)
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)

    # Prepare p-value plot
    anova_table = anova_table.reset_index().rename(columns={'index': 'variable'})
    anova_table = anova_table.sort_values('PR(>F)')
    y_labels = anova_table['variable']
    x_values = -np.log10(anova_table['PR(>F)'])
    colors = ['green' if p <= 0.05 else 'red' for p in anova_table['PR(>F)']]

    plt.figure(figsize=(10, 5))
    plt.barh(y_labels, x_values, color=colors)
    plt.axvline(-np.log10(0.05), color='black', linestyle='--', label='p = 0.05')
    plt.xlabel('-log10(p-value)')
    plt.title(f'ANCOVA variable p-values for best {result} model ({sort_by})')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plot_folder, f'{result}_best_model_pvalues.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved p-value plot for {result} to {plot_path}")

sort_by = 'bic'  # your criterion for best model

for result in list_of_results:
    # Select best model
    subset_models = results_df[results_df['result'] == result].sort_values(sort_by,
                                                                           ascending=(sort_by in ['aic', 'bic']))
    if subset_models.empty:
        continue
    best_model = subset_models.iloc[0]
    variables_in_model = [v.strip() for v in best_model['variables'].split(',')]

    # Remove Total dust for R_filter if needed
    if result == "R_filter_m^-1":
        variables_in_model = [v for v in variables_in_model if v != "Total dust (g)"]

    # Fit model
    formula = build_formula_ancova(result, variables_in_model)
    model = ols(formula, data=df).fit()

    # Prepare parity plot with matching lengths
    y_true = model.model.endog  # actual values used in fit
    y_pred = model.fittedvalues  # predicted values

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Parity plot for {result} (best model by {sort_by})')
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plot_folder, f'{result}_parity.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved parity plot for {result} to {plot_path}")

# --- Plot F-values for ANCOVA parameters ---
significance_level = 0.05

for result in list_of_results:
    subset = full_ancova_df[full_ancova_df['result'] == result].copy()

    if subset.empty:
        continue
    print(subset.head())
    # Sort by F-value
    subset = subset.sort_values('F', ascending=False)
    # Determine significance and colors
    subset['significant'] = subset['PR(>F)'] <= significance_level
    colors = ['green' if sig else 'red' for sig in subset['significant']]



    y_labels = subset['variable']
    x_values = subset['F']

    plt.figure(figsize=(10, 5))
    plt.barh(y_labels, x_values, color=colors)

    # Add significance line
    plt.axvline(x=0, color='black', linestyle='--')  # baseline at 0

    plt.xlabel('F-value')
    plt.title(f'ANCOVA F-values for {result}')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plot_folder, f'ANCOVA_Fvalues_{result}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved ANCOVA F-value plot for {result} to {plot_path}")
# --- Save top 5 models per metric to Excel ---
metrics = {
    'rsquared': True,      # True = descending (higher is better)
    'adj_rsquared': True,  # True = descending (higher is better)
    'aic': False,          # False = ascending (lower is better)
    'bic': False           # False = ascending (lower is better)
}

with pd.ExcelWriter(write_file, engine='openpyxl', mode='a') as writer:  # 'a' = append
    for metric, descending in metrics.items():
        top_models_metric = (
            results_df.sort_values(['result', metric], ascending=not descending)
            .groupby('result')
            .head(5)
        )

        # Reorder columns if needed
        ordered_cols = ['result', 'variables', 'rsquared', 'adj_rsquared', 'aic', 'bic',
                        'n_variables', 'p_value', 'f_stat_significant']
        top_models_metric = top_models_metric[ordered_cols]

        sheet_name = f'Top5_{metric}'
        top_models_metric.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Saved top 5 models by {metric} to sheet '{sheet_name}'")