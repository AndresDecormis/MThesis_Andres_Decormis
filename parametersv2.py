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

# Maximum electricity import/export per hour
p_imp_max               = 23                # Maximum electricity import capacity  [kW]
p_exp_max               = 23                # Maximum electricity export capacity  [kW]

# Hydrogen import/export per hour
h2_imp_max_val          = 10                # Maximum hydrogen import capacity per timestep [kg]
h2_exp_max_val          = 10                # Maximum hydrogen export capacity per timestep [kg]

# ----------------- Define modelling parameters -----------------
# Heat pump parameters                      # Source: T. Marti, M. Sulzer, ..., “Energieversorgung der schweiz bis 2050 zusammenfassung von ergebnissen und grundlagen,” 12 2022.
COP_hp              = 3.85                  # Coefficient of performance of heat pump [-] 
q_hp_max            = 15                # Maximum heat pump capacity [kW (heat)] 7.5 kW normal

# Thermal energy storage                # Source: T. Marti, M. Sulzer, ..., “Energieversorgung der schweiz bis 2050 zusammenfassung von ergebnissen und grundlagen,” 12 2022.
volume_tank         = 0.5               # Volume of the buffer tank [m^3] = 500 litres
heat_cap_tank       = 4.18              # Heat capacity of the buffer tank [kJ/kgK]
kJ_to_kWh           = 1/3600            # Conversion factor from kJ to kWh
density_water       = 1000              # Density of water [kg/m^3]
delta_temperature   = 30                # Temperature difference in the buffer tank [°C]
e_sto_max_tank_val  = volume_tank * heat_cap_tank * density_water * delta_temperature * kJ_to_kWh # Maximum storage capacity of the buffer tank [kWh]
stor_eff_tank       = 0.91              # Storage efficiency of the buffer tank [-]
standby_loss_tank   = 0.00971305        # Standby loss of the buffer tank [%ofkWh/h]
q_ts_in_max         = 0.25              # Maximum heat input to the tank [%ofkWh/h] - source: https://www.sciencedirect.com/science/article/pii/S0306261914007235?via%3Dihub
q_ts_out_max        = 0.25              # Maximum heat output to the tank [%ofkWh/h] - source: https://www.sciencedirect.com/science/article/pii/S0306261914007235?via%3Dihub

# --------- PWA Parameters ---------
# Electrolyser                  # Source: T. Marti, M. Sulzer, ..., “Energieversorgung der schweiz bis 2050 zusammenfassung von ergebnissen und grundlagen,” 12 2022.
l_el_max_val = 10                # Maximum electrolyser capacity [kW] - https://sweet-cross.ch/data/energy-tech-parameters/2024-02-27/
def electrolyser_function(x):
    """
    Function to calculate the electrolyser output based on the input capacity (normalised to 1)
    source: MOVE data (186 kW electrolyser)
    """
    return -0.1357  * (x**2) +  0.7171  * x #- 0.0088

# eff_el_h2   = 0.5719            # Electrolyser conversion efficiency [-]
eff_nom_el  = electrolyser_function(1) # Electrolyser nominal efficiency, at full load [-]
eff_el_th   = 0.95              # Electrolyser thermal efficiency [-] from total heat released what fraction is recoverable
h2_prod_max_val= l_el_max_val * eff_nom_el/ (HHV_H2) # Maximum hydrogen output capacity [kg/h] - https://sweet-cross.ch/data/energy-tech-parameters/2024-02-27/

# Fuel cell                                 # Source: T. Marti, M. Sulzer, ..., “Energieversorgung der schweiz bis 2050 zusammenfassung von ergebnissen und grundlagen,” 12 2022.
def fuel_cell_function(x):
    """
    Function to calculate the fuel cell output based on the input capacity (normalised to 1)
    source: https://www.sciencedirect.com/science/article/pii/S0196890421007408

    """
    return  -0.2377 * (x**2) + 0.6744 * x #- 0.0095        # TODO: Update this function

# eff_fc_h2   = 0.4279                        # Fuel cell conversion efficiency [-]
eff_nom_fc  = fuel_cell_function(1)         # Fuel cell nominal efficiency, at full load [-]
eff_fc_th   = 0.95                          # Fuel cell thermal efficiency [-] from total heat released what fraction is recoverable
p_fc_max_val= 5                             # Maximum fuel cell capacity [kW] - https://sweet-cross.ch/data/energy-tech-parameters/2024-02-27/
h2_fc_max_val= p_fc_max_val / (eff_nom_fc * HHV_H2) # Maximum hydrogen input capacity [kg/h] - https://sweet-cross.ch/data/energy-tech-parameters/2024-02-27/

# --------- End of PWA Parameters -------

# Compressor
l_co_max            = 1                # Maximum compressor capacity [kW]
T_in                = 70 + 273.15       # Inlet temperature in [K]
eff_co_isentropic   = 0.80              # Isentropic efficiency of the compressor [-] - source: ZQ
eff_co_mech         = 0.98              # Mechanical efficiency [-] - source: ZQ
eff_co_gen          = 0.96              # Electric generator efficiency [-] - source: ZQ
eff_co              = eff_co_isentropic * eff_co_mech * eff_co_gen         # Compressor efficiency [-]
P_out               = 60                # Outlet pressure in [bar]
P_in                = 30                # Inlet pressure in [bar]
comp_ratio          = (P_out / P_in)    # Pressure ratio
k_co_adiabatic      = (((R/M_H2)*T_in) * (GAMMA /(GAMMA-1)) * ((comp_ratio)**((GAMMA-1)/GAMMA)-1) / 3600) / eff_co # [kWh/kg] # Adiabatic compression coefficient - https://books.google.ch/books?id=irfr-Kd1nvEC&pg=PR3&source=gbs_selected_pages&cad=1#v=onepage&q&f=false
k_co_isothermal     = R*T_in*np.log(comp_ratio) / 3600 / M_H2 / eff_co      # Isothermal compression coefficient - https://books.google.ch/books?id=irfr-Kd1nvEC&pg=PR3&source=gbs_selected_pages&cad=1#v=onepage&q&f=false # [kWh/kg]
k_co_experimental   = 2.5470            # Experimental compression coefficient - source: Modelling, Identification and Control of a Renewable Hydrogen Production System for Mobility Applications Author(s): Laaksonlaita, Timo# [kWh/kg] - 900 bar
# Select the compression coefficient
k_co                = k_co_adiabatic
h2_max_comp         = l_co_max / k_co   # Maximum hydrogen compression capacity [kg/h]


# Hydrogen storage
vol_tank   = 0.85       # Volume in m^3, 850 litres per tank
press_tank = 60e5       # Pressure in Pa (1bar = 1e5 Pa)
T_tank     = 70 + 273.15# Temperature in K
num_tanks  = 4          # Number of tanks
h2_sto_eff = 0.938      # Hydrogen storage efficiency [-]
mass_h2    = num_tanks * (press_tank * vol_tank) / (R * T_tank) * M_H2 / 1000  # kg - results in approx 12 kg
h2_sto_max = mass_h2    # Maximum hydrogen storage capacity [kg]

# Battery storage
e_bat_cap_val= 30        # Maximum battery capacity [kWh]
bat_sto_eff  = 0.96      # Battery storage efficiency [-]
bat_max_ch   = 0.25      # Maximum battery charging capacity [%ofkWh/h] - source: https://www.sciencedirect.com/science/article/pii/S0306261923012278#b47
bat_max_dis  = 0.25      # Maximum battery discharging capacity [%ofkWh/h] - source: https://www.sciencedirect.com/science/article/pii/S0306261923012278#b47
e_bat_min    = 0.05      # Minimum battery capacity [%ofkWh]
e_bat_max    = 0.95      # Maximum battery capacity [%ofkWh]
self_dis_bat = 0.00054   # Self-discharge rate of battery [%ofkWh/h]

# PV parameters
pv_area      = 30       # Area of the PV panels [m^2] standard: 30m^2
pv_eff       = 0.17      # Efficiency of the PV panels [-]

# ----------------- Define consumer parameters -----------------
# Yearly electricity end-demands
lighting_yearly_demand           = 500      # Total lighting demand [kWh]
elec_appliances_yearly_demand    = 4500      # Total electric appliances demand [kWh]

# Heat demand
# Characteristics of the consumers
area_irradiance         = 40    # [m2] rooftop area of each consumer standard: 40m2
heat_input_conversion   = 0.04  # Qgain/Qsolar - Conversion factor from solar irradiance to heat input

# Consumers thermal inertia
R_cons_medium = 5.6        # Medium thermal resistance of consumer [°C/kW] - matches the actual energy demand of the nanoverbund pilot case
C_cons_medium = 14.71       # Medium thermal capacitance of consumer [kWh/°C] - bernardino
R_cons_high   = 7         # High thermal resistance of consumer [°C/kW] - 50% more
C_cons_high   = 14.71       # High thermal capacitance of consumer [kWh/°C] - 
R_cons_low    = 4         # Low thermal resistance of consumer [°C/kW] - 50% less
C_cons_low    = 14.71       # Low thermal capacitance of consumer [kWh/°C] - 

# OLD: Consumers thermal inertia
# R_cons_medium = 3.6         # Medium thermal resistance of consumer [°C/kW] - source ZJ
# C_cons_medium = 5.41        # Medium thermal capacitance of consumer [kWh/°C] - source ZJ
# R_cons_high   = 5           # High thermal resistance of consumer [°C/kW] - source https://www.sciencedirect.com/science/article/pii/S0196890408004780
# C_cons_high   = 7           # High thermal capacitance of consumer [kWh/°C] - source https://www.sciencedirect.com/science/article/pii/S0196890408004780
# R_cons_low    = 2           # Low thermal resistance of consumer [°C/kW] - 
# C_cons_low    = 4           # Low thermal capacitance of consumer [kWh/°C] - 


# Temperature set points   - Source: Indoor environmental input parameters for design and assessment of energy performance of buildings addressing indoor air quality, thermal environment, lighting and acoustics. ICS 91.140.01
T_cons_max_low  = 20.4      # Maximum temperature set point for low temperature flexibility consumer [°C]
T_cons_min_low  = 20.3        # Minimum temperature set point for low temperature flexibility consumer [°C]
T_cons_max_med  = 25        # Maximum temperature set point for medium temperature flexibility consumer [°C]
T_cons_min_med  = 20        # Minimum temperature set point for medium temperature flexibility consumer [°C]
T_cons_max_high = 25        # Maximum temperature set point for high temperature flexibility consumer [°C]
T_cons_min_high = 17.5      # Minimum temperature set point for high temperature flexibility consumer [°C]

