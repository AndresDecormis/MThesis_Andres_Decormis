# focs_shapley_quantification is a script that calculates the Fair Operational Cost Savings (FOCS) value 
# of energy technologies in a  multi-energy system. 


import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parametersv2 import *
import os
import additional_functions as af
import time
import itertools
import math
from functools import lru_cache

solve_counter = 0           # reset the counter

def main():
    """
    Main function to calculate FOCS values - it is iteratively called 
    """
    name_version = "Name-Scenario"                   # TODO: Name of the version of the results

    # Create results folders if they do not exist
    folder_version = f"results/00-ShapleyValues/{name_version}"
    if not os.path.exists(f"{folder_version}"):
        os.makedirs(f"{folder_version}")
    else:
        print(f"{folder_version} folder already exists.")
    # ---------------------------------------------------------
    st = time.process_time()    # get the start time
    
    # ---------------------------------------------------------
    # MAIN ALGORITHM
    # ---------------------------------------------------------
    # Players in the system - This can be changed to consider different sets of technologies
    n_players = 5
    name_players = ['PV', 'BES', 'TES', 'EL', 'FC']

    # Calculate the FOCS values
    tc_shapley_values, whv_shapley_values, wev_shapley_values           = calculate_shapley_value_memo(n_players, value_function)
    # Save the FOCS values in a csv file
    shapley_values_df = pd.DataFrame({
        "Total Cost": tc_shapley_values,
        "Heat Value": whv_shapley_values,
        "Electricity Value": wev_shapley_values
    })
    # Name the rows with the players' names
    shapley_values_df.index = name_players
    # Save the FOCS values in a csv file
    shapley_values_df.to_csv(f"{folder_version}/shapley_values.csv")

    # Plotting the Shapley values bar plot function
    bar_plotting_shapley_values(shapley_values_df, folder_version)

    # ---------------------------------------------------------

    et = time.process_time()    # get the end time
    res = et - st               # get execution time
    print('CPU Execution time:', res, 'seconds')
    print('Number of problems solved:', solve_counter)
    # Save computation time in a text file
    with open(f'{folder_version}/computation_time.txt', 'w') as f:
        f.write(f'CPU Execution time: {res} seconds')

def get_benchmark_values():
    # TODO: Define the benchmark values - Values from the scenario with no additional technologies (just HP)
    # These values can be obtained running the centralised_optimisation.py script only including a heat pump
    benchmark_total_cost    = 7417.0364 # CHF
    benchmark_whv_value     = 0.07875   # CHF/kWh
    benchmark_wev_value     = 0.31846   # CHF/kWh
    benchmark_values        = [benchmark_total_cost, benchmark_whv_value, benchmark_wev_value]
    return benchmark_values

def calculate_shapley_value_memo(n_players, value_func):
    """
    Calculates the Shapley values for 'n' decision points (number of players in our system) using the provided value function.
    Uses memoization to cache the value of the system for each state.
    :param n: Number of decision points (players)
    :param value_func: Function that calculates the value of the coalition for a given set of decisions
    :return: List of Shapley values for each decision point
    """
    
    # Memoized version of the system value function
    @lru_cache(None)  # Cache an unlimited number of values for the system's state
    def cached_value_func(state):
        return value_func(state)
    
    # Initialize the list to store the Shapley value for each decision point
    tc_shapley_values   = [0] * n_players
    whv_shapley_values  = [0] * n_players
    wev_shapley_values  = [0] * n_players
    
    # Iterate over all subsets of decision points
    for subset_size in range(n_players + 1):  # From size 0 to size n (all possible subsets)
        
        # Generate all possible subsets of this size
        for subset in itertools.combinations(range(n_players), subset_size):
            
            # Convert subset into a boolean tuple (True if in subset, False otherwise)
            subset_state = [False] * n_players  # Start with all decision points set to False
            for i in subset:
                subset_state[i] = True  # Set the decision points in the subset to True
            subset_state = tuple(subset_state)  # Convert list to tuple (so it's hashable)
            
            # For each decision point, calculate its contribution if it's not already in the subset
            for i in range(n_players):
                if i not in subset:
                    # Create a new state where decision point 'i' is added to the subset
                    subset_with_i = list(subset_state)
                    subset_with_i[i] = True
                    subset_with_i = tuple(subset_with_i)  # Convert list to tuple
                    
                    # Calculate the marginal contribution of decision point 'i'
                    values_with_i       = cached_value_func(subset_with_i)  # Use cached function
                    values_without_i    = cached_value_func(subset_state)   # Use cached function

                    # Extract the values from the tuple
                    tc_value_with_i     = values_with_i[0]     # Total cost value
                    whv_value_with_i    = values_with_i[1]     # Weighted average heat value
                    wev_value_with_i    = values_with_i[2]     # Weighted average electricity value
                    tc_value_without_i  = values_without_i[0]  # Total cost value
                    whv_value_without_i = values_without_i[1]  # Weighted average heat value
                    wev_value_without_i = values_without_i[2]  # Weighted average electricity value
                    
                    tc_marginal_contribution    = tc_value_with_i - tc_value_without_i
                    whv_marginal_contribution   = whv_value_with_i - whv_value_without_i
                    wev_marginal_contribution   = wev_value_with_i - wev_value_without_i
                    
                    # Calculate the weight for this marginal contribution
                    subset_len = len(subset)
                    weight = (math.factorial(subset_len) * math.factorial(n_players - subset_len - 1)) / math.factorial(n_players)
                    
                    # Add the weighted marginal contribution to the Shapley value for decision point 'i'
                    tc_shapley_values[i]    += weight * tc_marginal_contribution
                    whv_shapley_values[i]   += weight * whv_marginal_contribution
                    wev_shapley_values[i]   += weight * wev_marginal_contribution
    
    return tc_shapley_values, whv_shapley_values, wev_shapley_values

def value_function(decision_tuple: tuple):
    """
    Function to calculate the value of the coalition for a given set of decisions
    :param decision_list: tuple of decisions to be taken in the model
    :param benchmark_values: List of benchmark values to compare the coalition values with
    :return: Total cost value, weighted average heat value, weighted average electricity value
    """
    global solve_counter
    solve_counter += 1     # Increment the counter every time the function is computed (used to verify how many times the function is called)

    decision_list = list(decision_tuple)
    benchmark_total_cost, benchmark_whv_value, benchmark_wev_value = get_benchmark_values()
    coalition_total_cost, coalition_whv_value, coalition_wev_value = optimisation_function(decision_list)

    tc_value = benchmark_total_cost - coalition_total_cost
    whv_value = benchmark_whv_value - coalition_whv_value
    wev_value = benchmark_wev_value - coalition_wev_value

    return [tc_value, whv_value, wev_value]


def optimisation_function(decision_list: list):
    """
    Function to calculate the objective function of the optimisation problem for a given set of decisions
    :param decision_list: List of decisions to be taken in the model
    :return: Total cost, weighted average heat value, weighted average electricity
    """
    # -----------------------------------------------------------------------------------------------
    # Configuration of this run
    # -----------------------------------------------------------------------------------------------
    
    # Make every item in decision list False by default - then set the True values based on the decision list
    status_player = [False] * 5
    # Fix the code below

    for i, state in enumerate(decision_list):
        status_player[i] = state

    # -----------------------------------------------------------------------------------------------
    # Configurations of this run
    # -----------------------------------------------------------------------------------------------       
    import_price            = 'iwb'             # 'groupe_e' or 'bkw' or 'spot' or 'iwb'
    tariff_name             = 'power small'     # groupe_e: 'vario_plus', 'vario_grid', 'dt_plus' | bwk: 'green', 'blue', 'grey' | spot: 'spot' or 'plus_tariff' | iwb: 'power small', 'power small plus', 'power medium' or 'power medium plus'
    thermal_inertia         = 'Medium'          # Houses thermal inertia: 'Low', 'Medium', 'High', or 'Different'
    temp_flexibility        = "Medium"          # Use temperature flexibility in the model: "Low", "Medium", "High" or "Different"
    h2_price_scenario       = 'current'         # Hydrogen price scenario: 'current', 'future-low', 'future-high'
    electricity_consumption = True              # Do we take end-electricity consumption into account?
    use_pv                  = status_player[0]              # Use PV generation in the model
    use_thermal_storage     = status_player[2]              # Use thermal storage in the model
    use_battery_storage     = status_player[1]              # Use battery storage in the model
    use_electrolyser        = status_player[3]              # Use electrolyser in the model
    use_fuel_cell           = status_player[4]              # Use fuel cell in the model
    hydrogen_connection     = status_player[3] or status_player[4]            # Use hydrogen connection in the model if either electrolyser or fuel cell is used
    number_breakpoints      = 3                 # Number of breakpoints for the PWA function (Minimum is 0 - linear)
    # ----------------------------------------------------------------------------------------------
    constant_slack_cost     = 1e2               # [CHF/°C] Cost of exceeding max temperature. At 1e-1 it exceeds the temperature at some points in winter. At 1e0 it distorts heat value. At 1e6 it works well.
    # ----------------------------------------------------------------------------------------------

    # --------------------------------------------------------------
    # Define the parameters according to configuration of this run
    # --------------------------------------------------------------

    # ------------ Import electricity export price data ------------
    p_el_export             = af.get_spot_prices()                          # Electricity export price [CHF/kWh]
    if import_price         == 'groupe_e':
        p_el_import         = af.get_groupe_e_tariff_data(resolution='hourly', tariff=tariff_name)       # Electricity import price [CHF/kWh]
    elif import_price       == 'bkw':
        p_el_import         = af.get_bkw_tariff_data(tariff=tariff_name)       # Electricity import price [CHF/kWh]
    elif import_price       == 'iwb':
        p_el_import         = af.get_iwb_tariff_data(tariff=tariff_name)       # Electricity import price [CHF/kWh]
    elif import_price       == 'spot':
        if tariff_name      == 'spot':
            p_el_import     = af.get_spot_prices()                          # Electricity import price [CHF/kWh]
        elif tariff_name    == 'plus_tariff':
            p_el_import     = af.get_spot_prices() + spot_plus_tariff_1     # Electricity import price [CHF/kWh]
    else:
        raise ValueError("Error: Invalid type of electricity price")
    
    # ------------ Import other boundary conditions ------------
    T_amb                   = af.get_temperature_data()                     # Ambient temperature [°C]
    price_h2                = af.get_hydrogen_price(h2_price_scenario)      # Hydrogen price [CHF/kgH2]
    q_gain                  = af.get_heat_solar_gain_data()                 # Heat gain from sun irradiation [kW] - TODO: add real data based on solar irradiation
    light_prof, e_app_prof  = af.get_electricity_profile_demand(resolution="hourly") # Demand profiles for lighting and electric appliances [% of year]
    l_cons_demand_per_cons  = (light_prof * lighting_yearly_demand + e_app_prof * elec_appliances_yearly_demand)   # Total electricity demand from consumers [kWh]
    solar_irradiance        = af.get_solar_irradiance_data()                # Solar irradiance [kWh/m^2]
    cost_slack              = constant_slack_cost * np.ones(T)              # Cost of temperature slack variable

    # ------------ Select the thermal inertia ------------
    if thermal_inertia    == 'Low':
        R_cons1 = R_cons2 = R_cons3 = R_cons_low
        C_cons1 = C_cons2 = C_cons3 = C_cons_low
    elif thermal_inertia  == 'Medium':
        R_cons1 = R_cons2 = R_cons3 = R_cons_medium
        C_cons1 = C_cons2 = C_cons3 = C_cons_medium
    elif thermal_inertia  == 'High':
        R_cons1 = R_cons2 = R_cons3 = R_cons_high
        C_cons1 = C_cons2 = C_cons3 = C_cons_high
    elif thermal_inertia  == 'Different':
        R_cons1 = R_cons_low
        C_cons1 = C_cons_low
        R_cons2 = R_cons_medium
        C_cons2 = C_cons_medium
        R_cons3 = R_cons_high
        C_cons3 = C_cons_high
    else:
        raise ValueError("Error: Invalid type of thermal inertia")
    # Exponential decay terms for thermal inertia
    alpha_cons1= np.exp(-dt/(R_cons1*C_cons1))    # Decay term of consumer 1 [-]
    alpha_cons2= np.exp(-dt/(R_cons2*C_cons2))    # Decay term of consumer 2 [-]
    alpha_cons3= np.exp(-dt/(R_cons3*C_cons3))    # Decay term of consumer 3 [-]
    
    # ------------ Select the temperature flexibility of the consumers ------------
    if temp_flexibility == 'Low':
        T_cons1_min = T_cons2_min = T_cons3_min = T_cons_min_low
        T_cons1_max = T_cons2_max = T_cons3_max = T_cons_max_low
    elif temp_flexibility == 'Medium':
        T_cons1_min = T_cons2_min = T_cons3_min = T_cons_min_med
        T_cons1_max = T_cons2_max = T_cons3_max = T_cons_max_med
    elif temp_flexibility == 'High':
        T_cons1_min = T_cons2_min = T_cons3_min = T_cons_min_high
        T_cons1_max = T_cons2_max = T_cons3_max = T_cons_max_high
    elif temp_flexibility == 'Different':
        T_cons1_min = T_cons_min_low
        T_cons1_max = T_cons_max_low
        T_cons2_min = T_cons_min_med
        T_cons2_max = T_cons_max_med
        T_cons3_min = T_cons_min_high
        T_cons3_max = T_cons_max_high
    else:
        raise ValueError("Error: Invalid type of temperature flexibility")
    
    # ------------ Obtain the PWA functions for the electrolyser and fuel cell ------------
    x_values                        = np.linspace(0, 1, number_breakpoints+2)
    # ---- Electrolyser ----
    y_values_electrolyser           = electrolyser_function(x_values)
    PWA_el_slope, PWA_el_intercept  = af.get_PWA_lines(x_values, y_values_electrolyser)

    # ---- Fuel Cell ----
    y_values_fuel_cell              = fuel_cell_function(x_values)
    PWA_fc_slope, PWA_fc_intercept  = af.get_PWA_lines(x_values, y_values_fuel_cell)

    #--------------------------------------------------------------
    # Define the optimisation problem
    #--------------------------------------------------------------

    # Define the decision variables
    p_grid      = cp.Variable(T) # Net grid electricity [kW]
    p_imp       = cp.Variable(T,nonneg=True) # Import electricity [kW]
    p_exp       = cp.Variable(T,nonneg=True) # Export electricity [kW]
    h2_imp      = cp.Variable(T,nonneg=True) # Import hydrogen [kg]
    h2_exp      = cp.Variable(T,nonneg=True) # Export hydrogen [kg]

    # Define the dependant variables
    # ------ Power -------
    # Hydrogen
    p_h2        = cp.Variable(T) # net power from hydrogen system [kW]
    l_h2        = cp.Variable(T,nonneg=True) # load of hydrogen system [kW]
    l_el        = cp.Variable(T,nonneg=True) # load of electrolyser [kW]
    l_co        = cp.Variable(T,nonneg=True) # load of compressor [kW]
    p_fc        = cp.Variable(T,nonneg=True) # power generated from fuel cell [kW]
    # Heat pump
    l_hp        = cp.Variable(T,nonneg=True) # load of heat pump [kW]
    # Battery
    e_bat       = cp.Variable(T,nonneg=True) # energy stored in battery [kWh]
    p_bat_net   = cp.Variable(T) # net power from battery [kW]
    p_bat_ch    = cp.Variable(T,nonneg=True) # power charged to battery [kW]
    p_bat_dis   = cp.Variable(T,nonneg=True) # power discharged from battery [kW]
    # Load consumer
    l_cons1     = cp.Variable(T,nonneg=True) # load for electricity from consumer 1[kW]
    l_cons2     = cp.Variable(T,nonneg=True) # load for electricity from consumer 2[kW]
    l_cons3     = cp.Variable(T,nonneg=True) # load for electricity from consumer 3[kW]

    # ------- Heat -------
    # Hydrogen
    q_h2        = cp.Variable(T,nonneg=True) # heat used from hydrogen system [kW]
    q_el        = cp.Variable(T,nonneg=True) # heat used from electrolyser [kW]
    q_gen_el    = cp.Variable(T,nonneg=True) # heat generated from electrolyser [kW]
    q_was_el    = cp.Variable(T,nonneg=True) # heat wasted from electrolyser [kW]
    q_fc        = cp.Variable(T,nonneg=True) # heat used from fuel cell [kW]
    q_gen_fc    = cp.Variable(T,nonneg=True) # heat generated from fuel cell [kW]
    q_was_fc    = cp.Variable(T,nonneg=True) # heat wasted from fuel cell [kW]
    # Heat pump
    q_hp        = cp.Variable(T,nonneg=True) # heat used from heat pump [kW]
    # # Thermal storage (TES)
    q_ts_net    = cp.Variable(T) # net heat from thermal storage [kW]
    q_ts_in     = cp.Variable(T,nonneg=True) # heat input to buffer tank [kW]
    q_ts_out    = cp.Variable(T,nonneg=True) # heat output from buffer tank [kW]
    e_ts_sto    = cp.Variable(T,nonneg=True) # heat stored in buffer tank [kWh]
    # Heat consumer
    q_cons1     = cp.Variable(T,nonneg=True) # heat demand from consumer 1 [kW]
    q_cons2     = cp.Variable(T,nonneg=True) # heat demand from consumer 2 [kW]
    q_cons3     = cp.Variable(T,nonneg=True) # heat demand from consumer 3 [kW]

    # ------- Hydrogen -------
    h2_prod     = cp.Variable(T,nonneg=True) # hydrogen produced [kg]
    h2_fc       = cp.Variable(T,nonneg=True) # hydrogen used in fuel cell [kg]
    h2_imp      = cp.Variable(T,nonneg=True) # hydrogen imported [kg]
    h2_exp      = cp.Variable(T,nonneg=True) # hydrogen exported [kg]
    h2_sto      = cp.Variable(T,nonneg=True) # hydrogen stored [kg]

    # ------- Temperature -------
    # Consumer Indoor Temperature
    T_cons1     = cp.Variable(T) # temperature of consumer 1 [°C]
    T_cons2     = cp.Variable(T) # temperature of consumer 2 [°C]
    T_cons3     = cp.Variable(T) # temperature of consumer 3 [°C]

    # ------- Slack variables -------
    # Temperature slack variables
    T_cons1_slack = cp.Variable(T, nonneg=True)      # Slack variable for consumer 1 temperature
    T_cons2_slack = cp.Variable(T, nonneg=True)      # Slack variable for consumer 2 temperature
    T_cons3_slack = cp.Variable(T, nonneg=True)      # Slack variable for consumer 3 temperature
    
    # ------- Parameters -------
    # PV production
    p_pv        = cp.Parameter(T,nonneg=True) # power generated from PV: Parameter [kW]


    q_slack    = cp.Variable(T, nonneg=True) # Slack variable for heat balance [kW]

    # -------------------------------------------------------
    # ------- Conditional parameters --------
    # -------------------------------------------------------
    # Do we take electricity end-consumption into account?
    if electricity_consumption:
        l_each_cons  = l_cons_demand_per_cons     # Electricity end-demand from consumers
    elif not electricity_consumption:
        l_each_cons  = np.zeros(T)       # No electricity demand from consumers
    else:
        raise ValueError("Error: Invalid type of electricity consumption")
    
    # Do we use pv generation?
    if use_pv:
        pv_generation       = pv_area * pv_eff * solar_irradiance       # PV generation [kW]
        p_pv.value          = pv_generation                             # Use PV generation
    elif not use_pv:
        pv_generation       = 0
        p_pv.value          = np.zeros(T)                               # Do not use PV generation
    else:
        raise ValueError("Error: Invalid type of PV generation")

    # Do we use thermal storage?
    if use_thermal_storage:
        e_sto_max_tank      = e_sto_max_tank_val                         # Use thermal storage from parameters
    elif not use_thermal_storage:
        e_sto_max_tank      = 0                                          # Overriding parameters value not use thermal storage
    else:
        raise ValueError("Error: Invalid type of thermal storage")
    
    # Do we use battery storage?
    if use_battery_storage:
        e_bat_cap           = e_bat_cap_val                              # Use battery storage from parameters
    elif not use_battery_storage:
        e_bat_cap           = 0                                          # Overriding parameters value not use battery storage
    else:
        raise ValueError("Error: Invalid type of battery storage")

    # Do we use electrolyser?
    if use_electrolyser:
        l_el_max            = l_el_max_val                               # Use electrolyser from parameters
        h2_prod_max         = h2_prod_max_val                            # Use electrolyser from parameters
    elif not use_electrolyser:
        l_el_max            = 0                                          # Overriding parameters value not use electrolyser
        h2_prod_max         = 0                                          # Overriding parameters value not use electrolyser
    else:
        raise ValueError("Error: Invalid type of electrolyser")
    
    # Do we use fuel cell?
    if use_fuel_cell:
        p_fc_max            = p_fc_max_val                              # Use fuel cell from parameters
        h2_fc_max           = h2_fc_max_val                             # Use fuel cell from parameters
    elif not use_fuel_cell:
        p_fc_max            = 0                                          # Overriding parameters value not use fuel cell
        h2_fc_max           = 0                                          # Overriding parameters value not use fuel cell
    else:
        raise ValueError("Error: Invalid type of fuel cell")
    
    # Do we use hydrogen connection?
    if hydrogen_connection:
        h2_imp_max          = h2_imp_max_val                            # Use hydrogen connection from parameters
        h2_exp_max          = h2_exp_max_val                            # Use hydrogen connection from parameters
    elif not hydrogen_connection:
        h2_imp_max          = 0                                          # Overriding parameters value not use hydrogen connection
        h2_exp_max          = 0                                          # Overriding parameters value not use hydrogen connection
    else:
        raise ValueError("Error: Invalid type of hydrogen connection")

    # --------------------------------------------------------------
    #  Constraints 
    # --------------------------------------------------------------

    # ------ Global balance constraints -------
    # Heat balance constraint (constraint to derive heat value from)
    heat_balance_network            =  [q_h2 + q_hp ==  q_cons1 + q_cons2 + q_cons3 + q_ts_net + q_slack] 

    # Electricity balance constraint (constraint to derive electricity value from)
    electricity_balance_network     =  [p_pv == p_grid + p_bat_net + p_h2 + l_hp + l_cons1 + l_cons2 + l_cons3]

    # Hydrogen balance constraint
    h2_balance_network      = [p_h2         == l_h2 - p_fc,
                               l_h2         == l_el + l_co,
                               q_h2         == q_el + q_fc,]
    # Grid connections constraints 
    grid_connection         = [p_imp        <= p_imp_max,
                               p_exp        <= p_exp_max,
                               p_grid       == p_exp - p_imp]
    # Hydrogen connection constraints
    h2_connection           = [h2_imp       <= h2_imp_max,
                               h2_exp       <= h2_exp_max]

    # ------ Modelling components constraints -------
    # Electrolyser constraints
    electrolyser =    [l_el        <= l_el_max,
                       h2_prod     <= h2_prod_max,
                       q_gen_el    == (l_el - h2_prod * HHV_H2) * eff_el_th,
                       q_el        == q_gen_el - q_was_el]
    if use_electrolyser:
        for i in range(number_breakpoints+1):
            electrolyser += [h2_prod * HHV_H2    <=  PWA_el_intercept[i] * l_el_max +  PWA_el_slope[i] * l_el]
    # Fuel cell constraints
    fuel_cell =        [p_fc       <= p_fc_max,
                        h2_fc      <= h2_fc_max,
                        q_gen_fc   == (h2_fc * HHV_H2 - p_fc) * eff_fc_th,
                        q_fc       == q_gen_fc - q_was_fc]
    if use_fuel_cell:
        for i in range(number_breakpoints+1):
            fuel_cell += [p_fc  / (HHV_H2)   <=  PWA_fc_intercept[i] * h2_fc_max +  PWA_fc_slope[i] * h2_fc]
    # Compressor constraints
    compressor =       [l_co       <= l_co_max,
                        l_co       == k_co * h2_prod]
    # Hydrogen storage constraints
    h2_storage =   [h2_sto[0]      == h2_sto_max/2,
                    h2_sto[T-1]    == h2_sto_max/2,
                    h2_sto         <= h2_sto_max,
                    h2_sto[1:T]    == h2_sto[0:T-1] + h2_sto_eff * h2_prod[0:T-1] - (1/h2_sto_eff) * h2_fc[0:T-1] + h2_sto_eff * h2_imp[0:T-1] - (1/h2_sto_eff) * h2_exp[0:T-1]]
    # Heat pump constraints
    heat_pump =     [q_hp          <= q_hp_max,
                     q_hp          == COP_hp * l_hp]
    # Thermal energy storage constraints
    thermal_storage =  [e_ts_sto       <= e_sto_max_tank,
                        e_ts_sto[0]    == e_sto_max_tank/2,
                        e_ts_sto[T-1]  == e_sto_max_tank/2,
                        e_ts_sto[1:T]  == e_ts_sto[0:T-1] * (1 - standby_loss_tank) + stor_eff_tank * q_ts_in[0:T-1] - 1/stor_eff_tank * q_ts_out[0:T-1],
                        q_ts_in        <= q_ts_in_max * e_sto_max_tank,
                        q_ts_out       <= q_ts_out_max * e_sto_max_tank,
                        q_ts_net       == q_ts_in - q_ts_out]
    # Battery storage constraints
    battery_storage =  [e_bat[0]   == e_bat_cap/2,
                        e_bat[T-1] == e_bat_cap/2,
                        p_bat_ch   <= bat_max_ch * e_bat_cap,
                        p_bat_dis  <= bat_max_dis * e_bat_cap,
                        p_bat_net  == p_bat_ch - p_bat_dis,
                        e_bat      <= e_bat_max * e_bat_cap,
                        e_bat      >= e_bat_min * e_bat_cap,
                        e_bat[1:T] == e_bat[0:T-1] * (1 - self_dis_bat) + bat_sto_eff * p_bat_ch[0:T-1] - 1/bat_sto_eff * p_bat_dis[0:T-1]]

    # -------  Consumer constraints -------
    # Heat consumer 1 heat constraints
    consumer_1   = [l_cons1         == l_each_cons,
                    T_cons1         >= T_cons1_min,
                    T_cons1         <= T_cons1_max + T_cons1_slack,
                    T_cons1[0]      == (T_cons1_max + T_cons1_min) / 2,
                    T_cons1[1:T]    == alpha_cons1 * T_cons1[0:T-1] + (1-alpha_cons1) * (T_amb[0:T-1] + R_cons1 * (q_cons1[0:T-1] + q_gain[0:T-1]))]
    # Heat consumer 2 heat constraints
    consumer_2   = [l_cons2         == l_each_cons,
                    T_cons2         >= T_cons2_min,   
                    T_cons2         <= T_cons2_max + T_cons2_slack,
                    T_cons2[0]      == (T_cons2_max + T_cons2_min) / 2,
                    T_cons2[1:T]    == alpha_cons2 * T_cons2[0:T-1] + (1-alpha_cons2) * (T_amb[0:T-1] + R_cons2 * (q_cons2[0:T-1] + q_gain[0:T-1]))]
    # Heat consumer 3 heat constraints
    consumer_3   = [l_cons3         == l_each_cons,
                    T_cons3         >= T_cons3_min,
                    T_cons3         <= T_cons3_max + T_cons3_slack,
                    T_cons3[0]      == (T_cons3_max + T_cons3_min) / 2,
                    T_cons3[1:T]    == alpha_cons3 * T_cons3[0:T-1] + (1-alpha_cons3) * (T_amb[0:T-1] + R_cons3 * (q_cons3[0:T-1] + q_gain[0:T-1]))]
    

    # Define the objective function
    objective   = cp.Minimize(p_el_import.T @ p_imp - p_el_export.T @ p_exp + price_h2.T @ h2_imp - price_h2.T @ h2_exp + cost_slack.T @ T_cons1_slack + cost_slack.T @ T_cons2_slack + cost_slack.T @ T_cons3_slack)
    # Define the constraints
    constraints = heat_balance_network + electricity_balance_network + h2_balance_network + grid_connection + h2_connection + electrolyser + fuel_cell + compressor + h2_storage + heat_pump + thermal_storage + battery_storage + consumer_1 + consumer_2 + consumer_3
    # Create the problem
    problem     = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve(reoptimize=True,solver=cp.GUROBI, verbose=False, qcp=True)

    
    # ------ Main results -------
    total_cost              = problem.value - (cost_slack.T @ T_cons1_slack.value + cost_slack.T @ T_cons2_slack.value + cost_slack.T @ T_cons3_slack.value)  # scalar 
    heat_value              = - constraints[0].dual_value       # vector
    elec_value              = - constraints[1].dual_value       # vector
    average_heat_value      = np.average(heat_value)            # scalar
    average_elec_value      = np.average(elec_value)            # scalar
    weighted_avg_heat_value = heat_value.T @ (q_cons1.value + q_cons2.value + q_cons3.value) / np.sum(q_cons1.value + q_cons2.value + q_cons3.value)  # scalar
    weighted_avg_elec_value = elec_value.T @ (l_cons1.value + l_cons2.value + l_cons3.value) / np.sum(l_cons1.value + l_cons2.value + l_cons3.value)  # scalar


    return total_cost, weighted_avg_heat_value, weighted_avg_elec_value


def bar_plotting_shapley_values(shapley_values, folder_version):
    """
    Function to plot the Shapley values in a bar plot
    :param shapley_values: List of Shapley values for each decision point
    """
    af.configure_plots(style='fancy')

    name_players = shapley_values.index
    tc_shapley_values = shapley_values["Total Cost"]
    whv_shapley_values = shapley_values["Heat Value"]
    wev_shapley_values = shapley_values["Electricity Value"]

    # Create a bar plot of the Shapley values for total cost
    fig, ax = plt.subplots()
    ax.bar(name_players, tc_shapley_values, color =  "#002BFF")
    ax.set_ylabel('Shapley Value [CHF]')
    ax.set_title('Shapley Values for Total Cost')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{folder_version}/shapley_total_cost.pdf", format="pdf", bbox_inches="tight")

    # Create a bar plot of the Shapley values for heat value
    fig, ax = plt.subplots()
    ax.bar(name_players, whv_shapley_values, color =  "#BB0000")
    ax.set_ylabel('Shapley Value [CHF/kWh]')
    ax.set_title('Shapley Values for Heat Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{folder_version}/shapley_heat_value.pdf", format="pdf", bbox_inches="tight")

    # Create a bar plot of the Shapley values for electricity value
    fig, ax = plt.subplots()
    ax.bar(name_players, wev_shapley_values, color =  "#D6BA00")
    ax.set_ylabel('Shapley Value [CHF/kWh]')
    ax.set_title('Shapley Values for Electricity Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{folder_version}/shapley_electricity_value.pdf", format="pdf", bbox_inches="tight")


    # Stacked bar plot of the Shapley values, stacking by each technology, different bars for each shapley value
    fig, ax = plt.subplots()
    shapley_values_transposed = shapley_values.T
    # Create the stacked bar plot
    shapley_values_transposed.plot(kind='bar', stacked=True)
    # Add labels and title
    ax.set_xlabel('Components')
    ax.set_ylabel('Shapley Values')
    ax.set_title('Stacked Bar of Shapley Values per Player')
    ax.legend(title='Players')
    plt.xticks(rotation=45)
    plt.savefig(f"{folder_version}/shapley_stacked.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()  # Run the main function