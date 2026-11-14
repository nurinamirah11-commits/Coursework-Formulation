# Formulation Coursework 2026/2027
# Group 12 members:
# Nurin Amirah Binti Azmi (Student ID: 14539377)
# Ali Al Sarhani (Student ID: XXXXXXXX)
#
# Main objectives: 
# To mathematically model, simulate, and optimize the batch cooling crystallization process for the Optimus fungicide

# Library required for the codes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Experimental kinetics data of supersaturation (S), growth rate (G) and nucleation rate (J)
# Load data directly from the CSV file
data = pd.read_csv('SoF-Coursework-26-27-KineticData.csv')

# Print columns to see their exact names to extract the relevant data
# print(data.columns)

# Extract the relevant columns from the DataFrame
S_data = data['Supersaturation [-]'].to_numpy()              # Supersaturation
G_data = data['Growth rate [μm s^{-1}]'].to_numpy()          # Growth rate
J_data = data['Nucleation rate [s^{-1} kg^{-1}]'].to_numpy() # Nucleation rate

print("Number of data points:", len(S_data))

# Kinetic models for growth and nucleation rates
# Equations for growth rate (G) 
def G_model(S, k1, k2):
    S_safe = np.maximum(S, 1.0)  # Ensure S is at least 1 to avoid negative values because if S < 1, (S - 1) would be negative and raising a negative number to a fractional power can lead to a problem.
    return k1 * (S_safe - 1.0) ** k2

# Equations for nucleation rate (J) 
def J_model(S, A, B):
    S_safe = np.maximum(S, 1.0001)  # Ensure S is at least 1.0001 to avoid negative values, ln(1) = 0 and division by zero causes problems
    return A * np.exp(- B / (np.log(S_safe)) ** 2)

# Parameter estimation using curve_fit
# Initial guess for the parameters k1, k2 and A, B
initial_guess_G = [1.0, 1.0]
initial_guess_J = [1.0, 1.0]

# Fit the growth rate model to the data
popt_G, pcov_G = curve_fit(G_model, S_data, G_data, p0 = initial_guess_G, bounds = (0, np.inf)) # Adding bounds to declare that k1 and k2 are always positive, to avoid overflow errors
k1_est, k2_est = popt_G

# Fit the nucleation rate model to the data
popt_J, pcov_J = curve_fit(J_model, S_data, J_data, p0 = initial_guess_J, bounds = (0, np.inf)) # Adding bounds to declare that A and B are always positive, to avoid overflow errors
A_est, B_est = popt_J

print("\n=== PARAMETERS VALUES ===")
print(f"Estimated parameters for Growth Rate (G): k1 = {k1_est:.2f} μm s^-1, k2 = {k2_est:.2f}")
print(f"Estimated parameters for Nucleation Rate (J): A = {A_est:.2e} s^-1 kg^-1, B = {B_est:.2f}")

# Plot the experimental data and fitted models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

S_smooth = np.linspace(min(S_data), max(S_data), 200)

G_fit = G_model(S_smooth, *popt_G)
J_fit = J_model(S_smooth, *popt_J)

# Growth Rate Fit
ax1.plot(S_smooth, G_fit, color='red', linewidth=2, label='Fitted Model (G)')
ax1.scatter(S_data, G_data, color='blue', label='Experimental Data (G)', zorder=5)
ax1.set_xlabel('Supersaturation (S)', fontsize=11)
ax1.set_ylabel('Growth Rate (G)', fontsize=11)
ax1.set_title('Growth Rate Parameter Fit', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Nucleation Rate Fit
ax2.plot(S_smooth, J_fit, color='orange', linewidth=2, label='Fitted Model (J)')
ax2.scatter(S_data, J_data, color='green', label='Experimental Data (J)', zorder=5)
ax2.set_xlabel('Supersaturation (S)', fontsize=11)
ax2.set_ylabel('Nucleation Rate (J)', fontsize=11)
ax2.set_title('Nucleation Rate Parameter Fit', fontsize=12)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('Kinetic_Parameter_Fit.pdf', bbox_inches='tight')  # Save the figure as a PDF file
plt.show()