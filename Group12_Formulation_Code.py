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
k1 = 25.41      # μm s^-1
k2 = 1.32       # -
A = 1.40e+08     # s^-1 kg^-1 
B = 10.00        # - 

# Initial conditions
c_0 = 8.0          # g/kg (initial concentration of solute)
T_0 = 15.0         # °C (initial temperature)

# Seed parameters
L_bar = 100.0        # μm (average crystal length of seed), meaning that our gaussian distribution will be centered at 100 μm
sigma_1 = 10.0       # μm (standard deviation of seed crystal length), meaning that our gaussian distribution will have a spread of 10 μm
m_0 = 1.0            # g (initial mass of seed crystals)

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
t_max = 86400.0                       # s (maximum simulation time)
t_eval = np.linspace(0, t_max, 1000)  # s (time points for evaluation)

# Equations for solubility to calculate supersaturation later
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
def simulation_method_of_lines():
    # Initial seed distribution (because we have seed crystals)
    gauss = (1.0 / (sigma_1 * np.sqrt(2.0 * np.pi))) * np.exp(-((L_centers - L_bar) ** 2) / (2.0 * sigma_1 ** 2))
    f_0 = (m_0 / (rho_c * k_v * np.sum(L_centers ** 3 * gauss * delta_L))) * gauss
    y0_pbe = np.concatenate([[c_0], f_0])  # Initial state vector for PBE       

    def pbe_system(t, y):
        c = y[0]   # solute concentration
        f = y[1:]  # crystal size distribution (CSD)

        # Current variables at time t
        T = temperature(t)        # Get the current temperature
        c_star = solubility(T)    # Calculate the solubility at the current temperature
        S = c / c_star            # Calculate the supersaturation
        G = G_model(S, k1, k2)    # Growth rate
        J = J_model(S, A, B)      # Nucleation rate

        # Mass balance for solute concentration = second moment of the CSD
        mu_2 = np.sum(f * L_centers ** 2 * delta_L)  

        # Creates an empty array of zeros with the exact same length
        df_dt = np.zeros_like(f)

        # Equation df_dt for first bin, L = 0, the BC is f = J/G
        df_dt[0] = (J - G * f[0]) / delta_L

        # Equation df_dt for the rest of the bins, L > 0
        df_dt[1:] = - G * (f[1:] - f[:-1]) / delta_L

        # Solute mass balance coupled to second moment of the CSD
        dc_dt = - 3.0 * rho_c * k_v * G * mu_2
        return np.concatenate([[dc_dt], df_dt])

    sol = solve_ivp(pbe_system, (0, t_max), y0_pbe, t_eval = t_eval, method = 'RK45')
    return sol

# Method of Moments (MoM) to solve PBE
def simulation_method_of_moments():
    # Initial seed distribution (same initial seed bed as MoL)
    gauss = (1.0 / (sigma_1 * np.sqrt(2.0 * np.pi))) * np.exp(-((L_centers - L_bar) ** 2) / (2.0 * sigma_1 ** 2))
    f_0 = (m_0 / (rho_c * k_v * np.sum(L_centers ** 3 * gauss * delta_L))) * gauss

    # Initial moments mu_0, mu_1, mu_2, mu_3, mu_4 from initial CSD f_0
    mu_0_init = np.sum(f_0 * delta_L)
    mu_1_init = np.sum((L_centers ** 1) * f_0 * delta_L)
    mu_2_init = np.sum((L_centers ** 2) * f_0 * delta_L)
    mu_3_init = np.sum((L_centers ** 3) * f_0 * delta_L)
    mu_4_init = np.sum((L_centers ** 4) * f_0 * delta_L)

    y0_mom = np.array([c_0, mu_0_init, mu_1_init, mu_2_init, mu_3_init, mu_4_init])

    def mom_system(t, y):
        c = y[0]
        mu_0, mu_1, mu_2, mu_3, mu_4 = y[1], y[2], y[3], y[4], y[5]

        # Current variables at time t
        T = temperature(t)
        c_star = solubility(T)
        S = c / c_star
        G = G_model(S, k1, k2)
        J = J_model(S, A, B)

        # Moment equations (size-independent growth G, nucleation at L = 0)
        dmu_0_dt = J
        dmu_1_dt = G * mu_0
        dmu_2_dt = 2.0 * G * mu_1
        dmu_3_dt = 3.0 * G * mu_2
        dmu_4_dt = 4.0 * G * mu_3

        # Solute mass balance coupled to third moment mu_2
        dc_dt = -3.0 * rho_c * k_v * G * mu_2

        return np.array([dc_dt, dmu_0_dt, dmu_1_dt, dmu_2_dt, dmu_3_dt, dmu_4_dt])

    sol = solve_ivp(mom_system, (0, t_max), y0_mom, t_eval=t_eval, method='RK45')
    return sol

# Run Method of Lines
sol_pbe = simulation_method_of_lines()
t_pbe = sol_pbe.t
c_pbe = sol_pbe.y[0, :]
f_pbe = sol_pbe.y[1:, :]

crystal_mass_pbe = rho_c * k_v * np.sum((L_centers[:, None] ** 3) * f_pbe * delta_L, axis=0)
peak_size_init_pbe = L_centers[np.argmax(f_pbe[:, 0])]
peak_size_final_pbe = L_centers[np.argmax(f_pbe[:, -1])]

# Run Method of Moments
sol_mom = simulation_method_of_moments()
t_mom = sol_mom.t
c_mom = sol_mom.y[0, :]
mu_0_mom = sol_mom.y[1, :]
mu_1_mom = sol_mom.y[2, :]
mu_2_mom = sol_mom.y[3, :]
mu_3_mom = sol_mom.y[4, :]
mu_4_mom = sol_mom.y[5, :]

crystal_mass_mom = rho_c * k_v * mu_3_mom

# Number-weighted average length
num_mean_init_mom = mu_1_mom[0] / np.maximum(mu_0_mom[0], 1e-30)
num_mean_final_mom = mu_1_mom[-1] / np.maximum(mu_0_mom[-1], 1e-30)

# Volume-weighted average length
vol_mean_init_mom = mu_4_mom[0] / np.maximum(mu_3_mom[0], 1e-30)
vol_mean_final_mom = mu_4_mom[-1] / np.maximum(mu_3_mom[-1], 1e-30)

# Print results and comparison between the 2 methods
print(f"{'Variables':<28} | {'MoL':>12} | {'MoM':>12}")
print("-" * 58)
print(f"{'Initial Conc.   (g/kg)':<28} | {c_pbe[0]:>12.4f} | {c_mom[0]:>12.4f}")
print(f"{'Final Conc.     (g/kg)':<28} | {c_pbe[-1]:>12.4f} | {c_mom[-1]:>12.4f}")
print(f"{'Solute Consumed (g/kg)':<28} | {c_pbe[0] - c_pbe[-1]:>12.4f} | {c_mom[0] - c_mom[-1]:>12.4f}")
print("-" * 58)
print(f"{'Initial Seed Mass (g)':<28} | {crystal_mass_pbe[0]:>12.4f} | {crystal_mass_mom[0]:>12.4f}")
print(f"{'Final Crystal Mass (g)':<28} | {crystal_mass_pbe[-1]:>12.4f} | {crystal_mass_mom[-1]:>12.4f}")
print(f"{'Crystal Mass Gained (g)':<28} | {crystal_mass_pbe[-1] - crystal_mass_pbe[0]:>12.4f} | {crystal_mass_mom[-1] - crystal_mass_mom[0]:>12.4f}")

# Difference (MoM - MoL) and relative difference (%)
c_consumed_pbe = c_pbe[0] - c_pbe[-1]
c_consumed_mom = c_mom[0] - c_mom[-1]
mass_gain_pbe = crystal_mass_pbe[-1] - crystal_mass_pbe[0]
mass_gain_mom = crystal_mass_mom[-1] - crystal_mass_mom[0]

print("\n============== Differences (MoM vs. MoL) ==============")
difference = abs(c_mom[0] - c_pbe[0])
r = (difference / c_pbe[0] * 100.0) if abs(c_pbe[0]) > 1e-15 else 0.0
print(f"Initial Conc. Difference:            {difference:.4f} g/kg ({r:.2f}%)")

difference = abs(c_mom[-1] - c_pbe[-1])
r = (difference / c_pbe[-1] * 100.0) if abs(c_pbe[-1]) > 1e-15 else 0.0
print(f"Final Conc. Difference:              {difference:.4f} g/kg ({r:.2f}%)")

difference = abs(c_consumed_mom - c_consumed_pbe)
r = (difference / c_consumed_pbe * 100.0) if abs(c_consumed_pbe) > 1e-15 else 0.0
print(f"Solute Consumed Difference:          {difference:.4f} g/kg ({r:.2f}%)")

difference = abs(crystal_mass_mom[0] - crystal_mass_pbe[0])
r = (difference / crystal_mass_pbe[0] * 100.0) if abs(crystal_mass_pbe[0]) > 1e-15 else 0.0
print(f"Initial Seed Mass Difference:        {difference:.4f} g ({r:.2f}%)")

difference = abs(crystal_mass_mom[-1] - crystal_mass_pbe[-1])
r = (difference / crystal_mass_pbe[-1] * 100.0) if abs(crystal_mass_pbe[-1]) > 1e-15 else 0.0
print(f"Final Crystal Mass Difference:       {difference:.4f} g ({r:.2f}%)")

difference = abs(mass_gain_mom - mass_gain_pbe)
r = (difference / mass_gain_pbe * 100.0) if abs(mass_gain_pbe) > 1e-15 else 0.0
print(f"Crystal Mass Gained Difference:      {difference:.4f} g ({r:.2f}%)")

# MoL peak sizes and MoM number- & volume-weighted average lengths
print("\n================= SIZE METRICS =================")
print(f"MoL Initial Peak Size:               {peak_size_init_pbe:.4f} μm")
print(f"MoL Final Peak Size:                 {peak_size_final_pbe:.4f} μm")
print(f"\nMoM Number-Weighted Mean initial:  {num_mean_init_mom:.4f} μm")
print(f"MoM Number-Weighted Mean final:      {num_mean_final_mom:.4f} μm")
print(f"\nMoM Volume-Weighted Mean initial:  {vol_mean_init_mom:.4f} μm")
print(f"MoM Volume-Weighted Mean final:      {vol_mean_final_mom:.4f} μm")

# Plotting graph to see better the results, VERIFY LATER
# Time in hours for readable x-axis
t_hr_pbe = t_pbe / 3600.0
t_hr_mom = t_mom / 3600.0

# Solubility curve c*(T(t)) along the simulation time grid
T_profile = temperature(t_pbe)
c_star_profile = solubility(T_profile)

# Time-series mean sizes from MoM
num_mean_mom = mu_1_mom / np.maximum(mu_0_mom, 1e-30)
vol_mean_mom = mu_4_mom / np.maximum(mu_3_mom, 1e-30)

# Indices for initial, mid-batch, and final CSD snapshots
idx_init = 0
idx_mid = len(t_pbe) // 2
idx_final = -1

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Batch Cooling Crystallization: MoL vs. MoM Comparison", fontsize=14)

# 1. Solute concentration c(t) and solubility c*(T(t))
ax = axes[0, 0]
ax.plot(t_hr_pbe, c_pbe, label="c_pbe (MoL)", linewidth=2)
ax.plot(t_hr_mom, c_mom, "--", label="c_mom (MoM)", linewidth=2)
ax.plot(t_hr_pbe, c_star_profile, ":", label="Solubility c*(T)", linewidth=1.8)
ax.set_title("Solute Concentration vs. Time")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Concentration (g/kg)")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Total crystal mass: crystal_mass_pbe vs. crystal_mass_mom
ax = axes[0, 1]
ax.plot(t_hr_pbe, crystal_mass_pbe, label="crystal_mass_pbe (MoL)", linewidth=2)
ax.plot(t_hr_mom, crystal_mass_mom, "--", label="crystal_mass_mom (MoM)", linewidth=2)
ax.set_title("Total Crystal Mass vs. Time")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Crystal Mass (g)")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Crystal size distribution f_pbe(L, t) at initial, mid, and final times
ax = axes[1, 0]
ax.plot(L_centers, f_pbe[:, idx_init], label=f"t = {t_hr_pbe[idx_init]:.1f} h (initial)")
ax.plot(L_centers, f_pbe[:, idx_mid], label=f"t = {t_hr_pbe[idx_mid]:.1f} h (mid)")
ax.plot(L_centers, f_pbe[:, idx_final], label=f"t = {t_hr_pbe[idx_final]:.1f} h (final)")
ax.set_title("Crystal Size Distribution f_pbe (MoL)")
ax.set_xlabel("Crystal Length L_centers (μm)")
ax.set_ylabel("Distribution Density f_pbe")
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Number-weighted vs. volume-weighted average length (MoM)
ax = axes[1, 1]
ax.plot(t_hr_mom, num_mean_mom, label="Number-weighted (mu_1_mom / mu_0_mom)", linewidth=2)
ax.plot(t_hr_mom, vol_mean_mom, "--", label="Volume-weighted (mu_4_mom / mu_3_mom)", linewidth=2)
ax.set_title("Mean Crystal Size Evolution (MoM)")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Mean Length (μm)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()