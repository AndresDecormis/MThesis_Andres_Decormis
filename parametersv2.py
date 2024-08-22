import numpy as np

# Define global optimisation parameters
T                       = 8760              # Number of time periods [h]
dt                      = 1                 # Duration of each time period [h]
n_consumers             = 3                 # Number of consumers - nanoverbund pilot case

# Define constants
HHV_H2                  = 39.4              # Higher heating value of hydrogen [kWh/kg]
R                       = 8.3145            # Specific gas constant for gas in [J/(mol·K)]
M_H2                    = 2.016             # Molar mass of hydrogen in [g/mol]
GAMMA                   = 1.4               # Specific heat ratio for hydrogen [-]

# ----------------- Define network parameters -----------------
# Electricity price tariff on top of spot
spot_plus_tariff_1      = 0.13              # T&D addition for spot-based tariff [CHF/kWh] -- https://www.statista.com/statistics/1278749/electricity-price-breakdown-switzerland/

# ----------------- Define modelling parameters -----------------
# Heat pump parameters          # Source: T. Marti, M. Sulzer, ..., “Energieversorgung der schweiz bis 2050 zusammenfassung von ergebnissen und grundlagen,” 12 2022.
COP_hp      = 3.85              # Coefficient of performance of heat pump [-] 
q_hp_max    = 30                # Maximum heat pump capacity [kW (heat)]

# Buffer tank                           # Source: T. Marti, M. Sulzer, ..., “Energieversorgung der schweiz bis 2050 zusammenfassung von ergebnissen und grundlagen,” 12 2022.
vol_tank            = 1                 # Volume of the buffer tank [m^3]
stor_eff_tank       = 0.91              # Storage efficiency of the buffer tank [-]
standby_loss_tank   = 0.00971305        # Standby loss of the buffer tank [%ofkWh/h]
e_sto_max_tank      = 30                # Maximum buffer tank capacity [kWh (heat)]
q_ts_in_max         = 0.25              # Maximum heat input to the tank [%ofkWh/h] - source: https://www.sciencedirect.com/science/article/pii/S0306261914007235?via%3Dihub
q_ts_out_max        = 0.25              # Maximum heat output to the tank [%ofkWh/h] - source: https://www.sciencedirect.com/science/article/pii/S0306261914007235?via%3Dihub

# --------- PWA Parameters ---------
# Electrolyser                  # Source: T. Marti, M. Sulzer, ..., “Energieversorgung der schweiz bis 2050 zusammenfassung von ergebnissen und grundlagen,” 12 2022.
l_el_max    = 20                # Maximum electrolyser capacity [kW] - https://sweet-cross.ch/data/energy-tech-parameters/2024-02-27/
def electrolyser_function(x):
    """
    Function to calculate the electrolyser output based on the input capacity (normalised to 1)
    """
    return -0.066555  * (x**2) +  0.7099  * x + 0.005784314

eff_el_h2   = 0.63              # Electrolyser conversion efficiency [-]
eff_el_th   = 0.95              # Electrolyser thermal efficiency [-] from total heat released what fraction is recoverable

# Fuel cell                     # Source: T. Marti, M. Sulzer, ..., “Energieversorgung der schweiz bis 2050 zusammenfassung von ergebnissen und grundlagen,” 12 2022.
p_fc_max    = 20                # Maximum fuel cell capacity [kW] - https://sweet-cross.ch/data/energy-tech-parameters/2024-02-27/
h2_fc_max   = 0.82              # Maximum hydrogen input capacity [kg/h] - https://sweet-cross.ch/data/energy-tech-parameters/2024-02-27/
def fuel_cell_function(x):
    """
    Function to calculate the fuel cell output based on the input capacity (normalised to 1)
    """
    return -0.16555 * (x**2) + 0.6099 * x         # TODO: Update this function

eff_fc_h2   = 0.44              # Fuel cell conversion efficiency [-]
eff_fc_th   = 0.95              # Fuel cell thermal efficiency [-] from total heat released what fraction is recoverable
# --------- End of PWA Parameters -------

# Compressor
l_co_max    = 20                # Maximum compressor capacity [kW]
T_in        = 343               # Inlet temperature in [K]
eff_co      = 0.85              # Compressor efficiency [-]
P_out       = 900               # Outlet pressure in [bar]
P_in        = 30                # Inlet pressure in [bar]
comp_ratio  = (P_out / P_in)    # Pressure ratio
# Adiabatic compression coefficient
k_co_adiabatic      = R*T_in/((GAMMA-1)*eff_co) * ((comp_ratio)**((GAMMA-1)/GAMMA)-1) / M_H2 / 3600 # [kWh/kg]
# Isothermal compression coefficient
k_co_isothermal     = R*T_in*np.log(comp_ratio) / 3600 / M_H2 / eff_co                  # [kWh/kg]
# Experimental compression coefficient - source: Modelling, Identification and Control of a Renewable Hydrogen Production System for Mobility Applications Author(s): Laaksonlaita, Timo
k_co                = 2.5470                                                            # [kWh/kg] - 900 bar

# Hydrogen storage
vol_tank   = 82e-3      # Volume in m^3, 82 litres per tank
press_tank = 900e5      # Pressure in Pa (1bar = 1e5 Pa)
T_tank     = 288.15     # Temperature in K
num_tanks  = 2          # Number of tanks
h2_sto_eff = 0.99       # Hydrogen storage efficiency [-]
h2_imp_max = 10         # Maximum hydrogen import capacity per timestep [kg]
h2_exp_max = 10         # Maximum hydrogen export capacity per timestep [kg]
mass_h2    = num_tanks * (press_tank * vol_tank) / (R * T_tank) * M_H2 / 1000  # kg - results in approx 12 kg
h2_sto_max = mass_h2    # Maximum hydrogen storage capacity [kg]

# Battery storage
e_bat_cap   = 30        # Maximum battery capacity [kWh]
eff_bat     = 0.9       # Battery efficiency [-]
bat_sto_eff = 0.95      # Battery storage efficiency [-]
bat_max_ch  = 0.25      # Maximum battery charging capacity [%ofkWh/h] - source: https://www.sciencedirect.com/science/article/pii/S0306261923012278#b47
bat_max_dis = 0.25      # Maximum battery discharging capacity [%ofkWh/h] - source: https://www.sciencedirect.com/science/article/pii/S0306261923012278#b47
e_bat_min   = 0.05      # Minimum battery capacity [%ofkWh]
e_bat_max   = 0.95      # Maximum battery capacity [%ofkWh]
self_dis_bat= 0.00054   # Self-discharge rate of battery [%ofkWh/h]

# PV parameters
pv_area     = 20        # Area of the PV panels [m^2]
pv_eff      = 0.17      # Efficiency of the PV panels [-]

# ----------------- Define consumer parameters -----------------
# Yearly electricity end-demands
lighting_yearly_demand           = 1000      # Total lighting demand [kWh]
elec_appliances_yearly_demand    = 2000      # Total electric appliances demand [kWh]

# Heat demand
# Consumers thermal inertia
R_cons_medium = 3.6         # Medium thermal resistance of consumer [°C/kW] - source ZJ
C_cons_medium = 5.41        # Medium thermal capacitance of consumer [kWh/°C] - source ZJ
R_cons_high   = 5           # High thermal resistance of consumer [°C/kW] - source https://www.sciencedirect.com/science/article/pii/S0196890408004780
C_cons_high   = 7           # High thermal capacitance of consumer [kWh/°C] - source https://www.sciencedirect.com/science/article/pii/S0196890408004780
R_cons_low    = 2           # Low thermal resistance of consumer [°C/kW] - 
C_cons_low    = 4           # Low thermal capacitance of consumer [kWh/°C] - 
# Temperature set points   - Source: Indoor environmental input parameters for design and assessment of energy performance of buildings addressing indoor air quality, thermal environment, lighting and acoustics. ICS 91.140.01
T_cons_max_low  = 22.5      # Maximum temperature set point for low temperature flexibility consumer [°C]
T_cons_min_low  = 20        # Minimum temperature set point for low temperature flexibility consumer [°C]
T_cons_max_med  = 25        # Maximum temperature set point for medium temperature flexibility consumer [°C]
T_cons_min_med  = 20        # Minimum temperature set point for medium temperature flexibility consumer [°C]
T_cons_max_high = 25        # Maximum temperature set point for high temperature flexibility consumer [°C]
T_cons_min_high = 17.5      # Minimum temperature set point for high temperature flexibility consumer [°C]

