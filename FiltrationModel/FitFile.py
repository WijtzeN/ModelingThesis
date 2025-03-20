import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import FunctionFile as FF


# Define all models
def complete_model(t, Kb, J0):
    return np.array(FF.Complete(Kb, J0, t))


def standard_model(t, Ks, J0):
    return np.array(FF.Standard(Ks, J0, t))


def intermediate_model(t, Ki, J0):
    return np.array(FF.Intermediate(Ki, J0, t))


def cake_model(t, Kc, J0):
    return np.array(FF.Cake(Kc, J0, t))


def cake_complete_model(t, Kc, Kb, J0):
    return np.array(FF.CakeComplete(Kc, Kb, J0, t))


def cake_intermediate_model(t, Kc, Ki, J0):
    return np.array(FF.CakeIntermediate(Kc, Ki, J0, t))


def complete_standard_model(t, Kb, Ks, J0):
    return np.array(FF.CompleteStandard(Kb, Ks, J0, t))


def intermediate_standard_model(t, Ki, Ks, J0):
    return np.array(FF.IntermediateStandard(Ki, Ks, J0, t))


def cake_standard_model(t, Kc, Ks, J0):
    return np.array(FF.CakeStandard(Kc, Ks, J0, t))


def sigmoid_cake_complete_model(t, Kc, Kb, alpha, tf, b, J0):
    return np.array(FF.SigmoidCakeComplete(Kc, Kb, alpha, tf, b, J0, t))


def sigmoid_cake_intermediate_model(t, Kc, Ki, alpha, tf, b, J0):
    return np.array(FF.SigmoidCakeIntermediate(Kc, Ki, alpha, tf, b, J0, t))


def sigmoid_complete_standard_model(t, Kb, Ks, alpha, tf, b, J0):
    return np.array(FF.SigmoidCompleteStandard(Kb, Ks, alpha, tf, b, J0, t))


def sigmoid_intermediate_standard_model(t, Ki, Ks, alpha, tf, b, J0):
    return np.array(FF.SigmoidIntermediateStandard(Ki, Ks, alpha, tf, b, J0, t))


def sigmoid_cake_standard_model(t, Kc, Ks, alpha, tf, b, J0):
    return np.array(FF.SigmoidCakeStandard(Kc, Ks,  alpha, tf, b, J0, t))


# Load the Excel file
df = pd.read_excel('data1.xlsx')

# Assign the first column to t_data and the second to P_data
t_data = df.iloc[:, 0]
P_data = df.iloc[:, 2]

# Plot the raw data
plt.figure(figsize=(10, 8))
plt.scatter(t_data, P_data)
plt.ylabel(r"$\frac{P}{P0}$", rotation=0, fontsize=20, labelpad=20)
plt.xlabel("t", fontsize=20)
plt.title("Fitted Models", fontsize=25)
plt.show()

# Fixed parameter
J0 = 0.1

# Initial guessing parameters
g_Kb = 0.0003
g_Ki = 0.0003
g_Ks = 0.0003
g_Kc = 0.0003
g_alpha = 0.0003
g_tf = 4000
g_b = 0.04

# List of models to fit
models = [
    #("Complete Model", complete_model, ['Kb'], [g_Kb]),
    ("Standard Model", standard_model, ['Ks'], [g_Ks]),
    ("Intermediate Model", intermediate_model, ['Ki'], [g_Ki]),
    ("Cake Model", cake_model, ['Kc'], [g_Kc]),
    #("CakeComplete Model", cake_complete_model, ['Kc', 'Kb'], [g_Kc, g_Kb]),
    ("CakeIntermediate Model", cake_intermediate_model, ['Kc', 'Ki'], [g_Kc, g_Ki]),
    #("CompleteStandard Model", complete_standard_model, ['Kb', 'Ks'], [g_Kb, g_Ks]),
    ("IntermediateStandard Model", intermediate_standard_model, ['Ki', 'Ks'], [g_Ki, g_Ks]),
    ("CakeStandard Model", cake_standard_model, ['Kc', 'Ks'], [g_Kc, g_Ks]),
    ("SigmoidCakeComplete Model", sigmoid_cake_complete_model, ['Kc', 'Kb', 'alpha', 'tf', 'b'],
     [g_Kc, g_Kb, g_alpha, g_tf, g_b]),
    ("SigmoidCakeIntermediate Model", sigmoid_cake_intermediate_model,
     ['Kc', 'Ki', 'alpha', 'tf', 'b'], [g_Kc, g_Ki, g_alpha, g_tf, g_b]),
    ("SigmoidCompleteStandard Model", sigmoid_complete_standard_model,
     ['Kb', 'Ks', 'alpha', 'tf', 'b'], [g_Kb, g_Ks, g_alpha, g_tf, g_b]),
    ("SigmoidIntermediateStandard Model", sigmoid_intermediate_standard_model,
     ['Ki', 'Ks', 'alpha', 'tf', 'b'], [g_Ki, g_Ks, g_alpha, g_tf, g_b]),
    ("SigmoidCakeStandard Model", sigmoid_cake_standard_model, ['Kc', 'Ks', 'alpha', 'tf', 'b'],
     [g_Kc, g_Ks, g_alpha, g_tf, g_b])
]

# Initialize a table to store results
results = []

# Plot the original data
plt.figure(figsize=(10, 8))
plt.scatter(t_data, P_data, label='Data', color='black')

# Initialize the results table with the correct column order
columns = ["Function", "Kb", "Ks", "Ki", "Kc", "alpha", "tf", "b", "R²"]
results_table = pd.DataFrame(columns=columns)


# Loop over each model, fit, and plot
for model_name, model_func, param_names, param_guesses in models:
    # Define the boundaries for each parameter (you can modify these values)
    # The boundaries will be defined as tuples (lower_bound, upper_bound) for each parameter
    if model_name == "SigmoidCakeComplete Model":
        bounds = ([0, 0, 0, 0, 0], [1, 1, 1, 10000, 1])  # Example bounds
    elif model_name == "SigmoidCakeIntermediate Model":
        bounds = ([0, 0, 0, 0, 0], [1, 1, 1, 10000, 1])  # Example bounds
    elif model_name == "SigmoidCompleteStandard Model":
        bounds = ([0, 0, 0, 0, 0], [1, 1, 1, 10000, 1])  # Example bounds
    elif model_name == "SigmoidIntermediateStandard Model":
        bounds = ([0, 0, 0, 0, 0], [1, 1, 1, 10000, 1])  # Example bounds
    elif model_name == "SigmoidCakeStandard Model":
        bounds = ([0, 0, 0, 0, 0], [1, 1, 1, 10000, 1])  # Example bounds
    else:
        bounds = ([-np.inf] * len(param_names), [np.inf] * len(param_names))  # No bounds for other models

    try:
        # Fit the model to the data with initial guesses
        params, covariance = curve_fit(lambda t, *params: model_func(t, *params, J0), t_data, P_data, p0=param_guesses, bounds = bounds, maxfev=10000)

        # Generate the fitted curve
        fitted_curve = model_func(t_data, *params, J0)

        # Calculate R^2 (coefficient of determination)
        residuals = P_data - fitted_curve
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((P_data - np.mean(P_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Prepare the row to be inserted in the table
        result_row = {col: None for col in columns}  # Initialize a row with None values
        result_row["Function"] = model_name
        result_row["R²"] = r_squared

        # Fill the row with the fitted parameter values
        for param_name, param_value in zip(param_names, params):
            result_row[param_name] = param_value

        # Append the row to the results table
        results_table = pd.concat([results_table, pd.DataFrame([result_row])], ignore_index=True)
        plt.plot(t_data, fitted_curve, label=f"{model_name} (R² = {r_squared:.4f})", linewidth=1)

    except Exception as e:
        print(f"Error fitting {model_name}: {e}")

    # Save the results table to a CSV file
    results_table.to_csv('fitted_parameters.csv', index=False)
    print("Table saved to 'fitted_parameters.csv'")

# Add labels, title, and legend
plt.ylabel(r"$\frac{P}{P_0}$", rotation=0, fontsize=20, labelpad=20)
plt.xlabel("Time", fontsize=20)
plt.title("Fitted Models", fontsize=25)
plt.legend(loc='best', fontsize=10)
plt.show()

# Print the table
print(results_table)