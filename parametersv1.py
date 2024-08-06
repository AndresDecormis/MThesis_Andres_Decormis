import numpy as np


# Define global parameters
T                       = 8760              # Number of time periods [h]
dt                      = 1                 # Duration of each time period [h]
HHV_H2                  = 39.4              # Higher heating value of hydrogen [kWh/kg]

# Define models parameters
# Heat pump
COP_hp      = 3.5       # Coefficient of performance of heat pump [-] 
q_hp_max    = 30        # Maximum heat pump capacity [kW]

# Electrolyser
l_el_max    = 20        # Maximum electrolyser capacity [kW] - https://sweet-cross.ch/data/energy-tech-parameters/2024-02-27/
eff_el_h2   = 0.74      # Electrolyser conversion efficiency [-]
eff_el_th   = 0.92      # Electrolyser thermal efficiency [-]

# Fuel cell
p_fc_max    = 20        # Maximum fuel cell capacity [kW] - https://sweet-cross.ch/data/energy-tech-parameters/2024-02-27/
eff_fc_h2   = 0.62      # Fuel cell conversion efficiency [-]
eff_fc_th   = 0.92      # Fuel cell thermal efficiency [-]

# Compressor
l_co_max    = 20        # Maximum compressor capacity [kW]

# Compression factors
R       = 8.3145# Specific gas constant for gas in J/(mol·K)
M_H2    = 2.016  # Molar mass of hydrogen in g/mol
T_in    = 343   # Inlet temperature in [K]
gamma   = 1.4   # Specific heat ratio for air [-]
eff_co  = 0.85  # Compressor efficiency [-]
P_out   = 900   # Outlet pressure in [bar]
P_in    = 30    # Inlet pressure in [bar]
# Calculate the pressure ratio
compression_ratio = (P_out / P_in)
# Adiabatic compression coefficient
k_co_adiabatic  = R*T_in/((gamma-1)*eff_co) * ((compression_ratio)**((gamma-1)/gamma)-1) / M_H2 / 3600
# Isothermal compression coefficient
k_co_isothermal = R*T_in*np.log(compression_ratio) / 3600 / M_H2 / eff_co # Units in [kWh/kg]
# Experimental compression coefficient
k_co            = 2.5470    # Compressor conversion factor [kWh/kg] - 900 bar - Modelling, Identification and Control of a Renewable Hydrogen Production System for Mobility Applications Author(s): Laaksonlaita, Timo

# Heat demand
# Consumers parameters
R_cons_medium = 3.6       # Medium thermal resistance of consumer [°C/kW] - source ZJ
C_cons_medium = 5.41      # Medium thermal capacitance of consumer [kWh/°C] - source ZJ
R_cons_high   = 5         # High thermal resistance of consumer [°C/kW] - source https://www.sciencedirect.com/science/article/pii/S0196890408004780
C_cons_high   = 5.41        # High thermal capacitance of consumer [kWh/°C] - source https://www.sciencedirect.com/science/article/pii/S0196890408004780
R_cons_low    = 2         # Low thermal resistance of consumer [°C/kW] - 
C_cons_low    = 5.41         # Low thermal capacitance of consumer [kWh/°C] - 
# Temperature set points
T_cons1_max= 25     # Maximum temperature of consumer 1 [°C]
T_cons2_max= 25     # Maximum temperature of consumer 2 [°C]
T_cons3_max= 25     # Maximum temperature of consumer 3 [°C]
T_cons1_min= 20     # Minimum temperature of consumer 1 [°C]
T_cons2_min= 20     # Minimum temperature of consumer 2 [°C]
T_cons3_min= 20     # Minimum temperature of consumer 3 [°C]

# Hydrogen storage
vol_tank   = 82e-3  # Volume in m^3, 82 litres per tank
press_tank = 900e5  # Pressure in Pa
T_tank     = 288.15 # Temperature in K
num_tanks  = 2      # Number of tanks
# Calculate number of moles of hydrogen gas using the Ideal Gas Law
mass_h2    = num_tanks* (press_tank * vol_tank) / (R * T_tank) * M_H2 / 1000  # kg - results in approx 12 kg
# Resulting hydrogen storage parameters
h2_sto_max = mass_h2    # Maximum hydrogen storage capacity [kg]
h2_sto_eff = 0.99       # Hydrogen storage efficiency [-]
h2_imp_max = 10         # Maximum hydrogen import capacity per timestep [kg]
h2_exp_max = 10         # Maximum hydrogen export capacity per timestep [kg]
