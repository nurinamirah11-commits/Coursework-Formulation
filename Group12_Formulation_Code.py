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

# ==========================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================
# Fitted kinetic parameters from Phase 1
k1 = 1.25      
k2 = 1.50      
A  = 1.5e8     
B  = 18.5      

# Physical properties
rho_crystal = 1500.0          # Crystal density [kg/m^3]
k_v = np.pi / 4.0             # Cylindrical volumetric shape factor (~0.785)

# Discretization grid for crystal size (L)
L_max = 500.0                 # Maximum size [um]
dL = 1.0                      # Bin width [um]
L_edges = np.arange(0, L_max + dL, dL)
L_centers = (L_edges[:-1] + L_edges[1:]) / 2.0  # 500 bin center points [um]
N_bins = len(L_centers)

# Simulation runtime (24 hours = 86,400 seconds)
t_max = 86400.0  
t_eval = np.linspace(0, t_max, 200)

# ==========================================
# 2. SOLUBILITY & COOLING PROFILE FUNCTIONS
# ==========================================
def get_solubility(T):
    """Calculates saturation concentration C* [g/kg] at temperature T [°C]."""
    a_sol = 0.2
    b_sol = 5.0
    return a_sol * T + b_sol

def get_temperature_linear(t):
    """Case A: Standard linear cooling from 60°C to 15°C over 24h."""
    T_start = 60.0
    T_end = 15.0
    return T_start - (T_start - T_end) * (t / t_max)

def get_temperature_optimized(t):
    """Case B: Optimized non-linear cooling (slow early on to suppress massive initial fines)."""
    T_start = 60.0
    T_end = 15.0
    # Using a power-law trajectory for gentle initial cooling
    fraction = (t / t_max)**1.5
    return T_start - (T_start - T_end) * fraction

def calculate_kinetics(S):
    """Calculates Growth (G) and Nucleation (J) rates with safety bounds."""
    S_eff = np.maximum(S, 1.0)
    G = k1 * (S_eff - 1.0)**k2
    
    S_eff_nuc = np.maximum(S, 1.0001)
    J = A * np.exp(-B / (np.log(S_eff_nuc)**2))
    return G, J

# ==========================================
# 3. COMBINED PBE & METHOD OF MOMENTS SOLVER
# ==========================================
def run_crystallizer_simulation(cooling_profile_func):
    """
    Runs the coupled PBE (Method of Lines) + Method of Moments ODE system.
    State vector y layout:
    y[0]         = Solute Concentration (c)
    y[1]         = Moment 0 (m0 - count)
    y[2]         = Moment 1 (m1 - length)
    y[3]         = Moment 2 (m2 - surface area)
    y[4]         = Moment 3 (m3 - volume)
    y[5:]        = Size distribution bins f(L)
    """
    # Initial conditions (Table 1 Baseline)
    c_initial = 8.0
    T_initial = cooling_profile_func(0.0)
    
    # Seed distribution (Normal distribution centered at 100 um)
    mean_seed = 100.0
    std_seed = 10.0
    seed_mass = 1.0
    f_initial = (seed_mass / (rho_crystal * k_v * np.sum(L_centers**3 * dL))) * \
                np.exp(-((L_centers - mean_seed)**2) / (2 * std_seed**2))
    
    # Initial moments computed directly from seed distribution
    m0_init = np.sum(f_initial * dL)
    m1_init = np.sum(L_centers * f_initial * dL)
    m2_init = np.sum((L_centers**2) * f_initial * dL)
    m3_init = np.sum((L_centers**3) * f_initial * dL)
    
    y0 = np.concatenate([[c_initial, m0_init, m1_init, m2_init, m3_init], f_initial])
    
    def ode_system(t, y):
        c = y[0]
        m0, m1, m2, m3 = y[1], y[2], y[3], y[4]
        f = y[5:]
        
        # Thermodynamics & Kinetics
        T = cooling_profile_func(t)
        C_star = get_solubility(T)
        S = c / np.maximum(C_star, 1e-6)
        G, J = calculate_kinetics(S)
        
        # --- Method of Moments ODEs ---
        dm0_dt = J
        dm1_dt = G * m0
        dm2_dt = 2.0 * G * m1
        dm3_dt = 3.0 * G * m2
        
        # --- Finite Volume PBE ODEs ---
        df_dt = np.zeros_like(f)
        df_dt[0] = (J - G * f[0]) / dL
        df_dt[1:] = -G * (f[1:] - f[:-1]) / dL
        
        # --- Mass Balance (Solute Consumption) ---
        # Driven by the 3rd moment of the crystal population
        dc_dt = -3.0 * rho_crystal * k_v * G * m2
        
        return np.concatenate([[dc_dt, dm0_dt, dm1_dt, dm2_dt, dm3_dt], df_dt])

    # Run Solver
    solution = solve_ivp(ode_system, (0, t_max), y0, t_eval=t_eval, method='Radau')
    return solution

# ==========================================
# 4. EXECUTE BASELINE AND OPTIMIZED CASES
# ==========================================
print("Running Case A: Linear Cooling Simulation...")
sim_linear = run_crystallizer_simulation(get_temperature_linear)

print("Running Case B: Optimized Non-linear Cooling Simulation...")
sim_optimized = run_crystallizer_simulation(get_temperature_optimized)

# ==========================================
# 5. GENERATE REPORT-READY PLOTS
# ==========================================
fig = plt.figure(figsize=(14, 10))

# --- Plot 1: Solute Concentration Depletion ---
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(sim_linear.t / 3600, sim_linear.y[0], label='Linear Cooling', color='blue', lw=2)
ax1.plot(sim_optimized.t / 3600, sim_optimized.y[0], label='Optimized Cooling', color='orange', lw=2, linestyle='--')
ax1.set_title('Solute Concentration Profile $c(t)$')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Concentration (g/kg)')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# --- Plot 2: Mass Conservation Validation (PBE vs MoM 3rd Moment) ---
ax2 = fig.add_subplot(2, 2, 2)
# Calculate total solid volume from PBE bins: V_pbe = sum(L^3 * f * dL)
pbe_volume = np.sum((L_centers**3)[:, None] * sim_linear.y[5:, :], axis=0) * dL
mom_volume = sim_linear.y[4, :] # m3 from Method of Moments
ax2.plot(sim_linear.t / 3600, pbe_volume, label='PBE Bins Volume ($m_3$)', color='crimson', lw=2)
ax2.plot(sim_linear.t / 3600, mom_volume, label='MoM ODE Volume ($m_3$)', color='black', lw=1.5, linestyle=':')
ax2.set_title('Model Verification: Mass Conservation Check')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Third Moment / Volume Metric')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

# --- Plot 3: Final Crystal Size Distribution Comparison ---
ax3 = fig.add_subplot(2, 2, 3)
final_f_linear = sim_linear.y[5:, -1]
final_f_optimized = sim_optimized.y[5:, -1]
ax3.plot(L_centers, final_f_linear, label='Linear Cooling CSD', color='blue', lw=2)
ax3.plot(L_centers, final_f_optimized, label='Optimized Cooling CSD', color='green', lw=2)
ax3.set_title('Final Crystal Size Distribution at $t = 24$h')
ax3.set_xlabel('Crystal Size $L$ ($\mu$m)')
ax3.set_ylabel('Number Density $f(L)$')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend()

# --- Plot 4: Mean Crystal Size Progression ---
ax4 = fig.add_subplot(2, 2, 4)
mean_size_linear = sim_linear.y[2, :] / (sim_linear.y[1, :] + 1e-12)     # m1 / m0
mean_size_optimized = sim_optimized.y[2, :] / (sim_optimized.y[1, :] + 1e-12)
ax4.plot(sim_linear.t / 3600, mean_size_linear, label='Linear Cooling', color='blue', lw=2)
ax4.plot(sim_optimized.t / 3600, mean_size_optimized, label='Optimized Cooling', color='green', lw=2)
ax4.set_title('Mean Crystal Size $\\bar{L}$ Evolution')
ax4.set_xlabel('Time (hours)')
ax4.set_ylabel('Mean Size ($\mu$m)')
ax4.grid(True, linestyle='--', alpha=0.5)
ax4.legend()

plt.tight_layout()
plt.show()
print("All simulations and plots generated successfully!")