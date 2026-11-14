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
from scipy.integrate import solve_ivp

# Experimental kinetics data of supersaturation (S), growth rate (G) and nucleation rate (J)
# Load data directly from the CSV file
data = pd.read_csv('SoF-Coursework-26-27-KineticData.csv')

# 2. Print columns to see their exact names
print(df.columns)

