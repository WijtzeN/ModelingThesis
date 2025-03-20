import numpy as np
import pandas as pd
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
plt.title("intermediate", fontsize=25)
plt.show()

# Fixed parameter
J0 = 0.1

# Manually input the parameters
Kc =    0.000518277121045223
Ks =    0.000350164635585136
alpha = 0.000265636893317984
tf =    0
b =     0.00871083749102707

# Generate the model curve with the manually inputted parameters
fitted_curve = sigmoid_cake_standard_model(t_data, Kc, Ks, alpha, tf, b, J0)

# Plot the manually inputted model curve
plt.figure(figsize=(10, 8))
plt.plot(t_data, fitted_curve, label='SigmoidCakeStandard Model', color="red")
plt.scatter(t_data, P_data, label='Data', color='black')
plt.xlabel('Time', fontsize=20)
plt.ylabel(r"$\frac{P}{P0}$", rotation=0, fontsize=20, labelpad=20)
plt.legend(fontsize=12)
plt.title('All together', fontsize=25)
plt.show()

# Save the parameters and the curve results to a CSV file
results = {
    "Kc": Kc,
    "Ks": Ks,
    "alpha": alpha,
    "tf": tf
}
results_table = pd.DataFrame([results])
results_table.to_csv('manual_sigmoid_cake_standard_parameters.csv', index=False)
print("Table saved to 'manual_sigmoid_cake_standard_parameters.csv'")

# Print the results
print(results_table)
