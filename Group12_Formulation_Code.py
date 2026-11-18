# Formulation Coursework 2026/2027
# Group 12 members:
# Nurin Amirah Binti Azmi (Student ID: 14539377)
# Ali Al Sarhani (Student ID: XXXXXXXX)
#
# Main objectives: 
# To mathematically model, simulate, and optimize the batch cooling crystallization process for the Optimus fungicide

# Draft Part 2 : 
# Library required for the codes
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the parameters for the batch cooling crystallization process
# If combined with main (coding for parameters estimation), we can just call the parameters from 
# the main code instead of defining them here.
k_1 = 25.41      # μm s^-1
k_2 = 1.32       # -
A = 1.40e+08 # s^-1 kg^-1 
B = 10.00        # - 

# Initial conditions
c_0 = 8.0          # g/kg (initial concentration of solute)
T_0 = 15.0         # °C (initial temperature)

# Particle properties
rho_c = 1.11e-12 # g μm^-3 (particle density)
k_v = np.pi/4    # - (cylinderical shape factor)

# Discretization parameters for crystal size (L)
L_max = 500.0                                       # μm (maximum length of crystal)
delta_L = 1.0                                       # μm (bin length for discretization)
L_edges = np.arange(0, L_max + delta_L, delta_L)    # μm (edges of the bins)
L_centers = (L_edges[:-1] + L_edges[1:]) / 2        # μm (centers of the bins)
N_bins = len(L_centers)                             # Number of bins

# Simulation runtime parameters
t_max = 86400.0  # s (maximum simulation time)

# Equations for solubility
def solubility(T):
    a = 3.37                  # g/kg
    b = 0.0359                # °C^-1
    return a * np.exp(b * T)  # g/kg

# Equations for temperature
def temperature(t):
    T_final = 5.0                              # °C (final temperature that we chose)
    return T_0 - (T_0 - T_final) * (t / t_max) # assumption : linear cooling profile

# Calculate growth rate and solubility rate with parameters defined
# If merged with main coding, we can just call the J and G defined (J_model and G_model)
def G_model(S, k1, k2):
    S_safe = np.maximum(S, 1.0)  # Ensure S is at least 1 to avoid negative values because if S < 1, (S - 1) would be negative and raising a negative number to a fractional power can lead to a problem.
    return k1 * (S_safe - 1.0) ** k2

def J_model(S, A, B):
    S_safe = np.maximum(S, 1.0001)  # Ensure S is at least 1.0001 to avoid negative values, ln(1) = 0 and division by zero causes problems
    return A * np.exp(- B / (np.log(S_safe)) ** 2)

# Method of lines to solve PBE

