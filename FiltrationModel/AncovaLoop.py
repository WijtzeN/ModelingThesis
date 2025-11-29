import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from tabulate import tabulate
import itertools

# --- 1. Load Data ---
file = "summary_resultsloop.xlsx"
df = pd.read_excel(file, sheet_name="Summary", nrows=44)

def filter_dataframe(df, filters=None):
    if filters:
        # Build mask directly from df; each condition produces a series with df's index
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in filters.items():
            if isinstance(val, (list, tuple, set)):
                mask &= df[col].isin(val)
            else:
                mask &= df[col] == val
        return df.loc[mask]  # <--- Use .loc and mask built from df
    else:
        return df

# Usage Example
my_filters = {'Filter': 'Kayser PTFE',
              #'Substance': 'BWW40 Dust',
              #'Temp': 433.15,
              #'Flow': 5,
              #'Clean time': 120,
              #'Backwash Flow': 10,
              }    # AND condition on 'Substance' and 'Temp'
df = filter_dataframe(df, my_filters)
write_file = 'etaANCOVA.xlsx'

# Define your factors and covariates
controlled_factors = ['C(Substance)',
                      #'C(Filter)',
                      'C(Temp)',
                      'C(Flow)',
                      'C(Backwash_Flow)',
                      'C(Clean_time)'
                      ]

semi_controlled_covariates = [
    'Total dust (g)',
    #'Clean time',
    #'Flow',
    #'Temp',
    #'Backwash Flow',
#    "R_mem_flat1_N2_m^-1",
]

all_variables = controlled_factors + semi_controlled_covariates
controlled_names = [cf[2:-1] if cf.startswith('C(') and cf.endswith(')') else cf for cf in controlled_factors]

# Extract filter keys as they appear (assumed no C() wrapper here)
filters_keys = list(my_filters.keys())
filters_keys_extended = filters_keys + [f'C({key})' for key in filters_keys]
controlled_base = [cf[2:-1] if cf.startswith('C(') and cf.endswith(')') else cf for cf in controlled_factors]
combined_filter_keys = set(filters_keys_extended + controlled_base)
# Remove from all_variables any variable whose base name is in filters or controlled names
all_variables = [
    var for var in all_variables
    if var not in combined_filter_keys and f'C({var})' not in combined_filter_keys
]
print("Variables considered for ANCOVA:", all_variables)

list_of_results = [
    'eta',
    #'Rmem N2',
    #'Rmem steam',
    # 'alpha_m^-2',
    # 'R_peak_m^-1',
    # "R_mem_flat1_N2_m^-1",
    # "R_mem_flat2_N2plusSteam_m^-1",
    # "R_mem_flat3_N2_cleaned_m^-1",
    # "R_mem_flat4_N2plusSteam_cleaned_m^-1",
    # #'Rmem cleaned N2',
    # #'Rmem cleaned N2+Steam',
    # #'R frac',
    # #'M frac',
    # 'Dust in filter (g)',
    # # 'Ad. Res Clean N2',
    # # 'Ad. Res Clean N2 Steam'
    # 'New resistance N2',
    # 'New resistance N2+Steam',
]

# Suppose your DataFrame is called df_results
required_columns = list_of_results + all_variables  # all the columns you will use

required_columns_clean = [col[2:-1] if col.startswith("C(") and col.endswith(")") else col
                          for col in required_columns]

missing_cols = [col for col in required_columns_clean if col not in df.columns]

if missing_cols:
    print("Error: The following required columns are missing:")
    for col in missing_cols:
        print(f" - {col}")
    raise SystemExit("Stopping execution due to missing columns.")
else:
    print("All required columns are present.")

# Generate combinations: singles up to 9 variables
combinations_to_test = []
for r in range(1, 10):
    combinations_to_test.extend(list(itertools.combinations(all_variables, r)))

print(f"Testing {len(combinations_to_test)} combinations across {len(list_of_results)} results")


def build_formula_ancova(result, variables):
    """Build the ANCOVA formula with proper categorical handling"""
    formula_parts = []
    for var in variables:
        if var in controlled_factors:
            formula_parts.append(f'{var}')  # categorical vars may already include C()
        else:
            formula_parts.append(f'Q("{var}")')  # numerical covariates quoted
    formula = f'Q("{result}") ~ ' + ' + '.join(formula_parts)

    return formula

def test_model_combination(result, variables, df):
    """Test a specific combination of variables against a result"""

    try:
        formula = build_formula_ancova(result, variables)
        model = ols(formula, data=df).fit()
        return {
            'result': result,
            'variables': ', '.join(variables),
            'n_variables': len(variables),
            'rsquared': model.rsquared,
            'adj_rsquared': model.rsquared_adj,
            'f_pvalue': model.f_pvalue,
            'aic': model.aic,
            'bic': model.bic
        }, model
    except Exception as e:
        print(f"probleem {e}{result}{variables}")
        return None, None

# Store results
screening_results = []
all_model_params = []

# Test every combination against every result
for result in list_of_results:
    print(f"Testing result: {result}")
    for variable_comb in combinations_to_test:
        result_data, model = test_model_combination(result, variable_comb, df)
        if result_data:
            screening_results.append(result_data)
            # Save parameter details
            for param, coef in model.params.items():
                all_model_params.append({
                    "Result": result,
                    "Variables": ", ".join(variable_comb),
                    "Param": param,
                    "Coef": coef,
                    "StdErr": model.bse[param],
                    "t": model.tvalues[param],
                    "pvalue": model.pvalues[param],
                    "Adj R²": model.rsquared_adj,
                    "AIC": model.aic,
                    "BIC": model.bic,
                })

# Convert to DataFrames
results_df = pd.DataFrame(screening_results)
all_model_params_df = pd.DataFrame(all_model_params)

print(results_df.head())
# Add some metrics
results_df['f_stat_significant'] = results_df['f_pvalue'] < 0.05
results_df['model_complexity'] = results_df['n_variables'] / results_df['adj_rsquared'].clip(lower=0.01)
print(f"Completed {len(results_df)} model tests")

# Find best models for each result
best_models = results_df.sort_values(['result', 'adj_rsquared'], ascending=[True, False])
best_models = best_models.groupby('result').head(5)
best_models_bic = results_df.sort_values(['result', 'bic'], ascending=[True, True])
best_models_bic = best_models_bic.groupby('result').head(5)
best_models_bic_short = best_models_bic.groupby('result').head(1)

print(all_variables)

# Find most influential variables
variable_importance = []
for result in list_of_results:
    result_models = results_df[results_df['result'] == result]
    for variable in all_variables:
        base_var = variable[2:-1] if variable.startswith('C(') and variable.endswith(')') else variable
        #variable_models = result_models[result_models['variables'].str.contains(re.escape(base_var), regex=True)]
        variable_models = result_models[result_models['variables'].str.contains(base_var)]
        if len(variable_models) > 0:
            avg_improvement = variable_models['adj_rsquared'].mean()
            max_improvement = variable_models['adj_rsquared'].max()
            variable_importance.append({
                'result': result,
                'variable': variable,
                'avg_rsquared': avg_improvement,
                'max_rsquared': max_improvement,
                'times_significant': len(variable_models[variable_models['f_stat_significant']])
            })

variable_importance_df = pd.DataFrame(variable_importance)
overall_importance = variable_importance_df.groupby('variable').agg({
    'avg_rsquared': 'mean',
    'max_rsquared': 'max',
    'times_significant': 'sum'
}).sort_values('times_significant', ascending=False)
print("Overall Variable Importance:")
print(overall_importance)

# Summary reports
summary_report = []
for result in list_of_results:
    result_data = results_df[results_df['result'] == result]
    if len(result_data) > 0:
        best_model = result_data.nlargest(1, 'adj_rsquared').iloc[0]
        summary_report.append({
            'Result': result,
            'Best Model Variables': best_model['variables'],
            'Adj R²': f"{best_model['adj_rsquared']:.3f}",
            'AIC': f"{best_model['aic']:.1f}",
            'BIC': f"{best_model['bic']:.1f}",
            'Significant': 'Yes' if best_model['f_stat_significant'] else 'No',
            'P value': f"{best_model['f_pvalue']:.3e}",
            '# Sig Models': result_data['f_stat_significant'].sum()
        })
summary_report_aic = []
for result in list_of_results:
    result_data = results_df[results_df['result'] == result]
    if len(result_data) > 0:
        best_model = result_data.nsmallest(1, 'aic').iloc[0]
        summary_report_aic.append({
            'Result': result,
            'Best Model Variables': best_model['variables'],
            'Adj R²': f"{best_model['adj_rsquared']:.3f}",
            'AIC': f"{best_model['aic']:.1f}",
            'BIC': f"{best_model['bic']:.1f}",
            'Significant': 'Yes' if best_model['f_stat_significant'] else 'No',
            'P value': f"{best_model['f_pvalue']:.3e}",
            '# Sig Models': result_data['f_stat_significant'].sum()
        })
summary_report_bic = []
for result in list_of_results:
    result_data = results_df[results_df['result'] == result]
    if len(result_data) > 0:
        best_model = result_data.nsmallest(1, 'bic').iloc[0]
        summary_report_bic.append({
            'Result': result,
            'Best Model Variables': best_model['variables'],
            'Adj R²': f"{best_model['adj_rsquared']:.3f}",
            'AIC': f"{best_model['aic']:.1f}",
            'BIC': f"{best_model['bic']:.1f}",
            'P value': f"{best_model['f_pvalue']:.3e}",
            'Significant': 'Yes' if best_model['f_stat_significant'] else 'No',
            '# Sig Models': result_data['f_stat_significant'].sum()
        })


# Convert to DataFrames
summary_df = pd.DataFrame(summary_report)
summary_df_aic = pd.DataFrame(summary_report_aic)
summary_df_bic = pd.DataFrame(summary_report_bic)

# Save to Excel
with pd.ExcelWriter(write_file, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='All_Model_Tests', index=False)
    variable_importance_df.to_excel(writer, sheet_name='Variable_Importance', index=False)
    best_models.to_excel(writer, sheet_name='Best_Models_Summary', index=False)
    summary_df.to_excel(writer, sheet_name='Summary_Report_R2', index=False)
    summary_df_aic.to_excel(writer, sheet_name='Summary_Report_aic', index=False)
    summary_df_bic.to_excel(writer, sheet_name='Summary_Report_bic', index=False)
    all_model_params_df.to_excel(writer, sheet_name='All_Model_Params', index=False)
print("All results successfully saved to", write_file)
print("SUMMARY REPORT: BEST MODELS FOR EACH RESULT")
print("=" * 80)
print(tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False))
def plot_best_model_bic(df, best_models, overall_importance, controlled_factors, save_plots=False):
    for _, row in best_models.iterrows():
        result = row['result']
        model_vars = row['variables'].split(', ')
        # Most important variable (x-axis)
        importance_order = overall_importance.sort_values('avg_rsquared', ascending=False)
        most_important = next((var for var in importance_order.index if var in model_vars), model_vars[0])
        # Strip C() if needed
        if most_important.startswith('C(') and most_important.endswith(')'):
            most_important_col = most_important[2:-1].strip()
        else:
            most_important_col = most_important
        print(most_important_col)
        # Categorical variable
        cat_vars = [v for v in model_vars if v in controlled_factors]
        cat_var_col = None
        if cat_vars:
            cat_var = cat_vars[0]
            if cat_var.startswith('C(') and cat_var.endswith(')'):
                cat_var_col = cat_var[2:-1].strip()
            else:
                cat_var_col = cat_var
        # Build and fit formula
        formula_parts = [v if v in controlled_factors else f'Q("{v}")' for v in model_vars]
        formula = f'Q("{result}") ~ ' + ' + '.join(formula_parts)
        model = ols(formula, data=df).fit()
        # Predictions
        df_pred = df.copy()
        df_pred['predicted'] = model.predict(df_pred)
        # Plot
        plt.figure(figsize=(8,6))
        if cat_var_col:
            categories = df_pred[cat_var_col].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            for cat, color in zip(categories, colors):
                df_cat = df_pred[df_pred[cat_var_col] == cat]
                plt.scatter(df_cat[most_important_col], df_cat[result], label=f'{cat_var_col}={cat}', color=color)
                sorted_idx = np.argsort(df_cat[most_important_col])
                plt.plot(df_cat[most_important_col].iloc[sorted_idx], df_cat['predicted'].iloc[sorted_idx],marker='o', linestyle='None', color=color)
        else:
            plt.scatter(df_pred[most_important_col], df_pred[result], color='blue', label='Data')
            sorted_idx = np.argsort(df_pred[most_important_col])
            plt.plot(df_pred[most_important_col].iloc[sorted_idx], df_pred['predicted'].iloc[sorted_idx], marker='o', linestyle='None', color='red', label='Model')
        plt.xlabel(most_important_col)
        plt.ylabel(result)
        plt.title(f'Best BIC Model for {result}')
        if cat_var_col:
            plt.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{result}_best_model_bic.png', dpi=300)
        plt.show()
# Usage:
plot_best_model_bic(df, best_models_bic_short, overall_importance, controlled_factors)

# Store ANCOVA results: capturing ANOVA tables for each model tested
ancova_results = []

for result in list_of_results:
    print(f"Analyzing result: {result}")
    for variable_comb in combinations_to_test:
        try:
            formula = build_formula_ancova(result, variable_comb)
            model = ols(formula, data=df).fit()
            # Obtain Type II ANOVA table for more balanced test of factors
            anova_table = sm.stats.anova_lm(model, typ=3)

            # Extract summary stats for each variable in the model
            for var in variable_comb:
                # Match variable names in ANOVA table (handle categorical tokens here if present)
                match_vars = [x for x in anova_table.index if var in x]
                for mv in match_vars:
                    row = anova_table.loc[mv]
                    ancova_results.append({
                        'result': result,
                        'variables': ', '.join(variable_comb),
                        'variable': mv,
                        'sum_sq': row['sum_sq'],
                        'df': row['df'],
                        'F': row['F'],
                        'PR(>F)': row['PR(>F)']
                    })
        except Exception as e:
            # If model fails, skip
            continue

ancova_results_df = pd.DataFrame(ancova_results)

# Summarize: aggregate by variable across all models and results to assess importance
importance_summary = ancova_results_df.groupby(['result', 'variable']).agg({
    'sum_sq': 'mean',
    'F': 'mean',
    'PR(>F)': 'min'  # smallest p-value indicating strongest significance
}).reset_index()

importance_summary = importance_summary.sort_values(['result', 'F'], ascending=[True, False])

# # Save everything to Excel along with your previous results if desired
# with pd.ExcelWriter(write_file, engine='openpyxl') as writer:
#     ancova_results_df.to_excel(writer, sheet_name='ANCOVA_Results', index=False)
#     importance_summary.to_excel(writer, sheet_name='ANCOVA_Summary', index=False)

print("ANCOVA analysis complete. Results saved.")

# Display a summary table for top variables per result
for result in list_of_results:
    print(f"\nTop ANCOVA Effects for {result}")
    top_vars = importance_summary[importance_summary['result'] == result].head(10)
    print(tabulate(top_vars[['variable', 'F', 'PR(>F)']], headers='keys', tablefmt='github', showindex=False))


# Optional: Plot Top Effects for each result as bar charts
def plot_ancova_effects(df):
    for result in list_of_results:
        result_df = df[df['result'] == result].sort_values('F', ascending=True)
        colors = result_df['PR(>F)'].apply(lambda p: 'seagreen' if p < 0.05 else 'lightcoral')

        plt.figure(figsize=(8, 4))
        plt.barh(result_df['variable'], result_df['F'], color=colors)
        plt.xlabel('F-value (Effect Size)')
        plt.title(f'Full ANCOVA Model for {result}\n(green = significant, p<0.05)')
        plt.tight_layout()
        plt.show()

plot_ancova_effects(importance_summary)

print(result)
# Prepare formula for all variables
def build_full_formula(result, variables):
    formula_parts = []
    for var in variables:
        if var in controlled_factors:
            formula_parts.append(f'{var}')  # categorical factors may be already formatted with C()
        else:
            formula_parts.append(f'Q("{var}")')  # numeric covariates with quotes
    return f'Q("{result}") ~ ' + ' + '.join(formula_parts)

all_vars = all_variables  # controlled_factors + covariates

full_ancova_results = []

for result in list_of_results:
    print(f"Running full ANCOVA model for {result}")
    try:
        formula = build_full_formula(result, all_vars)
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)  # Type II ANOVA table

        for var in anova_table.index:
            row = anova_table.loc[var]
            full_ancova_results.append({
                'result': result,
                'variable': var,
                'sum_sq': row['sum_sq'],
                'df': row['df'],
                'F': row['F'],
                'PR(>F)': row['PR(>F)']
            })
    except Exception as e:
        print(f"Error in model for {result}: {e}")

full_ancova_df = pd.DataFrame(full_ancova_results)

# Summarize and sort by F value
for result in list_of_results:
    print(f"\nFull ANCOVA summary for {result}")
    res_df = full_ancova_df[full_ancova_df['result'] == result]
    display_df = res_df.sort_values('F', ascending=False)[['variable', 'F', 'PR(>F)']]
    print(tabulate(display_df, headers='keys', tablefmt='github', showindex=False))

def plot_pvalues_log(df):
    alpha = 0.05
    for result in list_of_results:
        result_df = df[df['result'] == result].sort_values('PR(>F)', ascending=True)

        plt.figure(figsize=(8, 4))
        plt.barh(result_df['variable'], -np.log10(result_df['PR(>F)']), color='skyblue')
        plt.axvline(-np.log10(alpha), color='red', linestyle='dotted', label=f'p = {alpha}')
        plt.xlabel('-log10(p-value)')
        plt.title(f'ANCOVA Significance for {result}')
        plt.legend()
        plt.tight_layout()
        plt.show()
plot_pvalues_log(full_ancova_df)


def plot_model_pvalues(models_df):
    """
    Plot top 5 models by BIC for each result, showing overall model p-value.

    models_df should contain columns:
        - 'result'
        - 'model_name' (or something identifying the model)
        - 'bic'
        - 'model_pvalue' (overall F-test p-value of the model)
    """
    alpha = 0.05

    for result in models_df['result'].unique():
        # Select models for this result
        result_df = models_df[models_df['result'] == result]

        # Pick top 5 models by lowest BIC
        top_models = result_df.nsmallest(5, 'bic')

        # Sort so bars are in a nice order
        top_models = top_models.sort_values('bic', ascending=False)

        # --- Plot ---
        plt.figure(figsize=(10, 4))
        plt.barh(top_models['variables'], -np.log10(top_models['f_pvalue']), color='skyblue')
        plt.axvline(-np.log10(0.05), color='red', linestyle='dotted', label='p = 0.05')
        plt.xlabel('-log10(p-value)')
        plt.title(f'Top 5 Models for {result} (by BIC)')
        plt.legend()
        plt.tight_layout()
        plt.show()
plot_model_pvalues(best_models)

def parity_plot_best_bic(df, best_models_bic_short, controlled_factors, save_plots=False):
    """
    Creates parity plots (predicted vs actual) for the best model by BIC for each result.
    """
    for _, row in best_models_bic_short.iterrows():
        result = row['result']
        model_vars = row['variables'].split(', ')

        # Build formula
        formula_parts = [v if v in controlled_factors else f'Q("{v}")' for v in model_vars]
        formula = f'Q("{result}") ~ ' + ' + '.join(formula_parts)

        # Fit model
        model = ols(formula, data=df).fit()

        # Predictions
        df_pred = df.copy()
        df_pred['predicted'] = model.predict(df_pred)

        # Parity plot
        plt.figure(figsize=(6,6))
        plt.scatter(df_pred[result], df_pred['predicted'], color='blue', edgecolor='k', alpha=0.7)
        max_val = max(df_pred[result].max(), df_pred['predicted'].max())
        min_val = min(df_pred[result].min(), df_pred['predicted'].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)  # 45-degree line
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f'Parity Plot: Best BIC Model for {result}')
        plt.grid(True)
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{result}_parity_best_bic.png', dpi=300)
        plt.show()

# Usage:
parity_plot_best_bic(df, best_models_bic_short, controlled_factors)