import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import FunctionFile as FF


def sigmoid_cake_standard_model(t, Kc, Ks, alpha, tf, b, J0):
    return np.array(FF.SigmoidCakeStandard(Kc, Ks, alpha, tf, b, J0, t))


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
plt.title("Fitted Model", fontsize=25)

# Fixed parameter
J0 = 0.1

# Initial guessing parameters
g_Kc = 0.0003
g_Ks = 0.00015
g_alpha = 0.0005
g_tf = 4000
g_b = 0.045

# Fit the sigmoid_cake_standard_model to the data with initial guesses
try:
    params, covariance = curve_fit(lambda t, Kc, Ks, alpha, tf, b: sigmoid_cake_standard_model(t, Kc, Ks, alpha, tf, b, J0),
                                   t_data, P_data, p0=[g_Kc, g_Ks, g_alpha, g_tf, g_b])

    # Generate the fitted curve
    fitted_curve = sigmoid_cake_standard_model(t_data, *params, J0)

    # Calculate R^2 (coefficient of determination)
    residuals = P_data - fitted_curve
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((P_data - np.mean(P_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Plot the fitted curve
    plt.plot(t_data, fitted_curve, label=f'SigmoidCakeStandard Model (R² = {r_squared:.4f})', color="red")
    plt.scatter(t_data, P_data, label='Data', color='black')
    plt.xlabel('Time', fontsize=20)
    plt.ylabel(r"$\frac{P}{P0}$", rotation=0, fontsize=20, labelpad=20)
    plt.legend(fontsize=12)
    plt.title('Fitted Model', fontsize=25)
    plt.show()

    # Save results
    results = {
        "Kc": params[0],
        "Ks": params[1],
        "alpha": params[2],
        "tf": params[3],
        "R²": r_squared
    }

    # Convert to DataFrame and save to CSV
    results_table = pd.DataFrame([results])
    results_table.to_csv('sigmoid_cake_standard_fitted_parameters.csv', index=False)
    print("Table saved to 'sigmoid_cake_standard_fitted_parameters.csv'")

    # Print the results
    print(results_table)

except Exception as e:
    print(f"Error fitting SigmoidCakeStandard Model: {e}")
