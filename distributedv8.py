
# distributedv8 is alligned with centralisedv8

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parametersv2 import *
import os
import additional_functions as af
import time
import itertools


def main():
        
    #-----
    # Values from centralised to plot
    min_cost_centralised    = 6448.775643 # Minimum cost for the system from the centralised optimisation

    # -----------------------------------------------------------------------------------------------
    # Configuration of this run
    af.configure_plots(style='fancy')
    #-------------------
    import_price            = 'iwb'             # 'groupe_e' or 'bkw' or 'spot' or 'iwb'
    tariff_name             = 'power small'     # groupe_e: 'vario_plus', 'vario_grid', 'dt_plus' | bwk: 'green', 'blue', 'grey' | spot: 'spot' or 'plus_tariff' | iwb: 'power small', 'power small plus', 'power medium' or 'power medium plus'
    thermal_inertia         = 'Medium'          # Houses thermal inertia: 'Low', 'Medium', 'High', or 'Different'
    temp_flexibility        = "Medium"          # Use temperature flexibility in the model: "Low", "Medium", "High" or "Different"
    h2_price_scenario       = 'current'         # Hydrogen price scenario: 'current', 'future-low', 'future-high'
    other_notes             = '_NoTESLowcost-NewRemove'                # Notes to be added on the folder version name
    electricity_consumption = True              # Do we take end-electricity consumption into account?
    use_pv                  = True              # Use PV generation in the model
    use_thermal_storage     = False              # Use thermal storage in the model
    use_battery_storage     = True              # Use battery storage in the model
    use_electrolyser        = True              # Use electrolyser in the model
    use_fuel_cell           = True              # Use fuel cell in the model
    hydrogen_connection     = True              # Use hydrogen connection in the model
    number_breakpoints      = 3                 # Number of breakpoints for the PWA function (Minimum is 0 - linear)
    # -----------------
    # Type of algorithm for optimisation
    algorithm_type          = 'ADMM'            # 'ADMM' or 'LR' (Lagrangian Relaxation - ignoring augmented lagrangian)
    #-------------------
    save_results            = True
    plot_results            = False
    save_images             = True
    #-------------------
    name_version            = f"distributedv8" \
        f"-{algorithm_type}" \
        f"-{import_price}_{tariff_name}" \
        f"-{thermal_inertia}_TI" \
        f"-{electricity_consumption}_EC" \
        f"-{use_pv}_PV" \
        f"-{use_battery_storage}_BES" \
        f"-{use_thermal_storage}_TES" \
        f"-{use_electrolyser}_EL" \
        f"-{use_fuel_cell}_FC" \
        f"-{temp_flexibility}_TF" \
        f"-{hydrogen_connection}_H2" \
        f"-{h2_price_scenario}_H2Price" \
        f"-{number_breakpoints}_PWA" \
        f"{other_notes}" # Convention: scriptname-tariff-thermal_inertia
    name_version            = name_version.replace(" ", "_")
    # ----------------------------------------------------------------------------------------------
    # A high cost affects convergence
    constant_slack_cost = 1e0 # [CHF/°C] At 1e-1 it exceeds the temperature at some points in winter. At 1e0 it does not.NOTE: This is a constant cost for the slack variable, a very high value (>1e3) may affect the convergence of the algorithm
    #-------------------
    # Plotting time frames
    weekly_time_frames = [
    ('2023-01-01', '2023-01-07'),  # Winter week
    ('2023-04-01', '2023-04-07'),  # Spring week
    ('2023-07-01', '2023-07-07'),  # Summer week
    ('2023-10-01', '2023-10-07')   # Autumn week
    ]
    daily_time_frames = [
    ('2023-01-01', '2023-01-02'),  # Winter day
    ('2023-04-01', '2023-04-02'),  # Spring day
    ('2023-07-01', '2023-07-02'),  # Summer day
    ('2023-10-01', '2023-10-02')   # Autumn day
    ]
    # -----------------

    # ---------------------------------------------------------
    # Parameter values for the ADMM algorithm
    # ---------------------------------------------------------
    # Penalty parameter values
    # rho_heat_values      = [0.001, 0.002,0.005,0.01,0.05,0.1]     # Initial value of the penalty parameter
    # rho_elec_values      = [0.001, 0.002,0.005,0.01,0.05,0.1]     # Initial value of the penalty parameter
    # rho_heat_values      = [0.005,0.01,0.05]     # Initial value of the penalty parameter
    # rho_elec_values      = [0.005,0.01,0.05]     # Initial value of the penalty parameter
    rho_heat_values      = [0.05]     # Initial value of the penalty parameter
    rho_elec_values      = [0.05]     # Initial value of the penalty parameter
    max_iter        = 3            # Maximum number of iterations
    tolerance_primal= 1e-4          # Tolerance of the primal residual
    tolerance_dual  = 1e-4          # Tolerance of the dual residual
    n_heat_agents   = 6             # Number of agents - h2, hp, cons1, cons2, cons3, TES
    n_elec_agents   = 7             # Number of agents - h2, hp, cons1, cons2, cons3, BES, Grid
    # ---------------------------------------------------------

    # Create results folders if they do not exist
    folder_version = f"results/{name_version}"
    if not os.path.exists(f"{folder_version}"):
        os.makedirs(f"{folder_version}")
    else:
        print(f"{folder_version} folder already exists.")

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
    # Start the optimistaion over multiple penalty parameters
    #--------------------------------------------------------------
    rho_combinations = list(itertools.product(rho_heat_values, rho_elec_values))

    for rho_heat_fixed, rho_elec_fixed in rho_combinations:

        #--------------------------------------------------------------
        # Define the optimisation problem
        #--------------------------------------------------------------
        # Define the penalty parameter
        rho_heat    = cp.Parameter(nonneg=True) # Penalty parameter for heat
        rho_elec    = cp.Parameter(nonneg=True) # Penalty parameter for electricity

            
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
        # Thermal storage (TES)
        q_ts_net    = cp.Variable(T) # net heat from thermal storage [kW]
        q_ts_in     = cp.Variable(T,nonneg=True) # heat input to thermal storage [kW]
        q_ts_out    = cp.Variable(T,nonneg=True) # heat output from thermal storage [kW]
        e_ts_sto    = cp.Variable(T,nonneg=True) # heat stored in thermal storage [kWh]
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

        # Define parameters to be exchanged and updated through iterations
        lambda_heat_fixed       = cp.Parameter(T)
        lambda_elec_fixed       = cp.Parameter(T)
        # Grid parameters
        p_grid_fixed            = cp.Parameter(T)
        # Hydrogen system parameters
        q_h2_fixed              = cp.Parameter(T)
        p_h2_fixed              = cp.Parameter(T)
        # Heat pump parameters
        q_hp_fixed              = cp.Parameter(T)
        l_hp_fixed              = cp.Parameter(T)
        # Battery storage parameters
        p_bat_net_fixed         = cp.Parameter(T)
        # Thermal storage parameters
        q_ts_net_fixed          = cp.Parameter(T)
        # Heat consumer parameters
        q_cons1_fixed           = cp.Parameter(T)
        q_cons2_fixed           = cp.Parameter(T)
        q_cons3_fixed           = cp.Parameter(T)
        l_cons1_fixed           = cp.Parameter(T)
        l_cons2_fixed           = cp.Parameter(T)
        l_cons3_fixed           = cp.Parameter(T)
        # Average values to be exchanged
        q_average               = cp.Parameter(T)
        p_average               = cp.Parameter(T)

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
            l_el_max            = l_el_max_val                              # Use electrolyser from parameters
        elif not use_electrolyser:
            l_el_max            = 0                                          # Overriding parameters value not use electrolyser
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

        # ---------------------------------------------------------------
        # Define functions to build optimisation problems for each agent
        # ---------------------------------------------------------------
        # Hydrogen system
        def hydrogen_system_optimisation():
            """
            hydrogen_system_optimisation defines the optimisation problem for the hydrogen system
            """
            # Define the cost function
            cost_h2             =  price_h2.T @ h2_imp - price_h2.T @ h2_exp 

            # Define the constraints
            hydrogen_network_balance = [p_h2        == l_h2 - p_fc,
                                        l_h2        == l_el + l_co,
                                        q_h2        == q_el + q_fc]
            # Hydrogen connection constraints
            h2_connection            = [h2_imp        <= h2_imp_max,
                                        h2_exp        <= h2_exp_max]
            # Electrolyser constraints
            electrolyser             = [l_el        <= l_el_max,
                                        q_gen_el    == (l_el - h2_prod * HHV_H2) * eff_el_th,
                                        q_gen_el    == q_was_el + q_el]
            for i in range(number_breakpoints+1):
                electrolyser += [h2_prod * HHV_H2 / l_el_max   <=  PWA_el_intercept[i] +  PWA_el_slope[i] * l_el / l_el_max]
            # Fuel cell constraints
            fuel_cell =                [p_fc       <= p_fc_max,
                                        q_gen_fc   == (h2_fc * HHV_H2 - p_fc) * eff_fc_th,
                                        q_gen_fc   == q_was_fc + q_fc]
            for i in range(number_breakpoints+1):
                fuel_cell += [p_fc  / (h2_fc_max * HHV_H2)   <=  PWA_fc_intercept[i] +  PWA_fc_slope[i] * h2_fc / h2_fc_max]
            # Compressor constraints
            compressor               = [l_co       <= l_co_max,
                                        l_co       == k_co * h2_prod]
            # Hydrogen storage constraints
            h2_storage               = [h2_sto[0]      == h2_sto_max/2,
                                        h2_sto[T-1]    == h2_sto_max/2,
                                        h2_sto         <= h2_sto_max,
                                        h2_sto[1:T]    == h2_sto[0:T-1] + h2_sto_eff * h2_prod[0:T-1] - (1/h2_sto_eff) * h2_fc[0:T-1] + h2_sto_eff * h2_imp[0:T-1] - (1/h2_sto_eff) * h2_exp[0:T-1]]
            # Define augmented lagrangian
            lagr_h2             = cost_h2 - lambda_heat_fixed.T @ q_h2 + lambda_elec_fixed.T @ p_h2
            if algorithm_type == 'ADMM':
                augm_term_h2        = rho_heat /2 * cp.sum_squares(n_heat_agents * q_average + q_h2_fixed - q_h2) \
                                    + rho_elec /2 * cp.sum_squares(n_elec_agents * p_average - p_h2_fixed + p_h2)
            elif algorithm_type == 'LR':
                augm_term_h2        = 0
            augm_lagr_h2        = lagr_h2 + augm_term_h2
            constraints_h2      = hydrogen_network_balance + h2_connection + electrolyser + fuel_cell + compressor + h2_storage
            objective_h2        = cp.Minimize(augm_lagr_h2)
            # Create the problem
            problem_h2          = cp.Problem(objective_h2, constraints_h2)
            return problem_h2

        # Heat pump
        def heat_pump_optimisation():
            """
            heat_pump_optimisation defines the optimisation problem for the heat pump
            """
            # Define the cost function
            cost_hp             = 0
            # Define heat pump constraints
            heat_pump           =  [q_hp   <= q_hp_max,
                                    q_hp   == COP_hp * l_hp]
            # Define augmented lagrangian
            lagr_hp             = cost_hp - lambda_heat_fixed.T @ q_hp + lambda_elec_fixed.T @ l_hp
            if algorithm_type == 'ADMM':
                augm_term_hp        = rho_heat / 2 * cp.sum_squares(n_heat_agents * q_average + q_hp_fixed - q_hp) \
                                    + rho_elec / 2 * cp.sum_squares(n_elec_agents * p_average - l_hp_fixed + l_hp)
            elif algorithm_type == 'LR':
                augm_term_hp        = 0
            augm_lagr_hp        = lagr_hp + augm_term_hp
            constraints_hp      = heat_pump
            objective_hp        = cp.Minimize(augm_lagr_hp)
            # Create the problem
            problem_hp          = cp.Problem(objective_hp, constraints_hp)
            return problem_hp

        # Heat consumer 1
        def consumer_1_optimisation():
            """
            consumer_1_optimisation defines the optimisation problem for consumer 1
            """
            cost_cons1      = cost_slack.T @ T_cons1_slack
            # Define load demand constraint
            load_consumer_1 = [l_cons1 == l_each_cons]
            # Define heat consumer heat constraints
            consumer_1   = [T_cons1         >= T_cons1_min,
                            T_cons1         <= T_cons1_max + T_cons1_slack,
                            T_cons1[0]      == (T_cons1_max + T_cons1_min) / 2,
                            T_cons1[1:T]    == alpha_cons1 * T_cons1[0:T-1] + (1-alpha_cons1) * (T_amb[0:T-1] + R_cons1 * (q_cons1[0:T-1] + q_gain[0:T-1]))]
            # Define augmented lagrangian
            lagr_cons1              = cost_cons1 + lambda_heat_fixed.T @ q_cons1 + lambda_elec_fixed.T @ l_cons1
            if algorithm_type == 'ADMM':
                augm_term_cons1         = rho_heat / 2 * cp.sum_squares(n_heat_agents * q_average - q_cons1_fixed + q_cons1) \
                                        + rho_elec / 2 * cp.sum_squares(n_elec_agents * p_average - l_cons1_fixed + l_cons1)
            elif algorithm_type == 'LR':
                augm_term_cons1         = 0
            augm_lagr_cons1         = lagr_cons1 + augm_term_cons1
            constraints_cons1       = load_consumer_1 + consumer_1
            objective_cons1         = cp.Minimize(augm_lagr_cons1)
            # Create the problem
            problem_cons1 = cp.Problem(objective_cons1, constraints_cons1)
            return problem_cons1

        # Heat consumer 2
        def consumer_2_optimisation():
            """
            consumer_2_optimisation defines the optimisation problem for consumer 2
            """
            cost_cons2      = cost_slack.T @ T_cons2_slack
            # Define load demand constraint
            load_consumer_2 = [l_cons2 == l_each_cons]
            # Define heat consumer heat constraints
            consumer_2   = [T_cons2         >= T_cons2_min,
                            T_cons2         <= T_cons2_max + T_cons2_slack,
                            T_cons2[0]      == (T_cons2_max + T_cons2_min) / 2,
                            T_cons2[1:T]    == alpha_cons2 * T_cons2[0:T-1] + (1-alpha_cons2) * (T_amb[0:T-1] + R_cons2 * (q_cons2[0:T-1] + q_gain[0:T-1]))]
            # Define augmented lagrangian
            lagr_cons2              = cost_cons2 + lambda_heat_fixed.T @ q_cons2 + lambda_elec_fixed.T @ l_cons2
            if algorithm_type == 'ADMM':
                augm_term_cons2         = rho_heat / 2 * cp.sum_squares(n_heat_agents * q_average - q_cons2_fixed + q_cons2) \
                                    + rho_elec / 2 * cp.sum_squares(n_elec_agents * p_average - l_cons2_fixed + l_cons2)
            elif algorithm_type == 'LR':
                augm_term_cons2         = 0
            augm_lagr_cons2         = lagr_cons2 + augm_term_cons2
            constraints_cons2       = load_consumer_2 + consumer_2
            objective_cons2         = cp.Minimize(augm_lagr_cons2)
            # Create the problem
            problem_cons2 = cp.Problem(objective_cons2, constraints_cons2)
            return problem_cons2
        
        # Heat consumer 3
        def consumer_3_optimisation():
            """
            consumer_3_optimisation defines the optimisation problem for consumer 3
            """
            cost_cons3      = cost_slack.T @ T_cons3_slack
            # Define load demand constraint
            load_consumer_3 = [l_cons3 == l_each_cons]
            # Define heat consumer heat constraints
            consumer_3   = [T_cons3         >= T_cons3_min,
                            T_cons3         <= T_cons3_max + T_cons3_slack,
                            T_cons3[0]      == (T_cons3_max + T_cons3_min) / 2,
                            T_cons3[1:T]    == alpha_cons3 * T_cons3[0:T-1] + (1-alpha_cons3) * (T_amb[0:T-1] + R_cons3 * (q_cons3[0:T-1] + q_gain[0:T-1]))]
            # Define augmented lagrangian
            lagr_cons3              = cost_cons3 + lambda_heat_fixed.T @ q_cons3 + lambda_elec_fixed.T @ l_cons3
            if algorithm_type == 'ADMM':
                augm_term_cons3         = rho_heat / 2 * cp.sum_squares(n_heat_agents * q_average - q_cons3_fixed + q_cons3) \
                                        + rho_elec / 2 * cp.sum_squares(n_elec_agents * p_average - l_cons3_fixed + l_cons3)
            elif algorithm_type == 'LR':
                augm_term_cons3         = 0
            augm_lagr_cons3         = lagr_cons3 + augm_term_cons3
            constraints_cons3       = load_consumer_3 + consumer_3
            objective_cons3         = cp.Minimize(augm_lagr_cons3)
            # Create the problem
            problem_cons3 = cp.Problem(objective_cons3, constraints_cons3)
            return problem_cons3
        
        # Thermal energy storage
        def thermal_storage_optimisation():
            """
            thermal_storage_optimisation defines the optimisation problem for the thermal storage
            """
            # Define the cost function
            cost_ts         = 0
            # Define the constraints
            thermal_storage =  [e_ts_sto       <= e_sto_max_tank,
                                e_ts_sto[0]    == e_sto_max_tank/2,
                                e_ts_sto[T-1]  == e_sto_max_tank/2,
                                e_ts_sto[1:T]  == e_ts_sto[0:T-1] * (1 - standby_loss_tank) + stor_eff_tank * q_ts_in[0:T-1] - 1/stor_eff_tank * q_ts_out[0:T-1],
                                q_ts_in        <= q_ts_in_max * e_sto_max_tank,
                                q_ts_out       <= q_ts_out_max * e_sto_max_tank,
                                q_ts_net       == q_ts_in - q_ts_out]
            # Define augmented lagrangian
            lagr_ts             = cost_ts + lambda_heat_fixed.T @ q_ts_net
            if algorithm_type == 'ADMM':
                augm_term_ts        = rho_heat / 2 * cp.sum_squares(n_heat_agents * q_average - q_ts_net_fixed + q_ts_net)
            elif algorithm_type == 'LR':
                augm_term_ts        = 0
            augm_lagr_ts        = lagr_ts + augm_term_ts
            constraints_ts      = thermal_storage
            objective_ts        = cp.Minimize(augm_lagr_ts)
            # Create the problem
            problem_ts          = cp.Problem(objective_ts, constraints_ts)
            return problem_ts
        
        # Battery energy storage
        def battery_storage_optimisation():
            """
            battery_storage_optimisation defines the optimisation problem for the battery storage
            """
            # Define the cost function
            cost_bat        = 0
            # Define the constraints
            battery_storage =  [e_bat[0]   == e_bat_cap/2,
                                e_bat[T-1] == e_bat_cap/2,
                                p_bat_ch   <= bat_max_ch * e_bat_cap,
                                p_bat_dis  <= bat_max_dis * e_bat_cap,
                                p_bat_net  == p_bat_ch - p_bat_dis,
                                e_bat      <= e_bat_max * e_bat_cap,
                                e_bat      >= e_bat_min * e_bat_cap,
                                e_bat[1:T] == e_bat[0:T-1] * (1 - self_dis_bat) + bat_sto_eff * p_bat_ch[0:T-1] - 1/bat_sto_eff * p_bat_dis[0:T-1]]
            # Define augmented lagrangian
            lagr_bat             = cost_bat + lambda_elec_fixed.T @ p_bat_net
            if algorithm_type == 'ADMM':
                augm_term_bat        = rho_elec / 2 * cp.sum_squares(n_elec_agents * p_average - p_bat_net_fixed + p_bat_net)
            elif algorithm_type == 'LR':
                augm_term_bat        = 0
            augm_lagr_bat        = lagr_bat + augm_term_bat
            constraints_bat      = battery_storage
            objective_bat        = cp.Minimize(augm_lagr_bat)
            # Create the problem
            problem_bat          = cp.Problem(objective_bat, constraints_bat)
            return problem_bat

        # Grid connection operation
        def grid_connection_optimisation():
            """
            battery_storage_optimisation defines the optimisation problem for the battery storage
            """
            # Define the cost function
            cost_grid             = p_el_import.T @ p_imp - p_el_export.T @ p_exp
            # Define the constraints
            grid_connection       = [p_imp        <= p_imp_max,
                                     p_exp        <= p_exp_max,
                                     p_grid       == p_exp - p_imp]
            # Define augmented lagrangian
            lagr_grid             = cost_grid + lambda_elec_fixed.T @ p_grid
            if algorithm_type   == 'ADMM':
                augm_term_grid        = rho_elec / 2 * cp.sum_squares(n_elec_agents * p_average - p_grid_fixed + p_grid)
            elif algorithm_type == 'LR':
                augm_term_grid        = 0
            augm_lagr_grid        = lagr_grid + augm_term_grid
            constraints_grid      = grid_connection
            objective_grid        = cp.Minimize(augm_lagr_grid)
            # Create the problem
            problem_grid          = cp.Problem(objective_grid, constraints_grid)
            return problem_grid
        
        # -------------------------------------------------
        # Define functions to return the optimal values
        # -------------------------------------------------
        # Hydrogen system
        def hydrogen_system_optimal_value(problem_h2):
            """
            hydrogen_system_optimal_value return the optimal heat produced by the hydrogen system
            """
            # Solve the problem
            problem_h2.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_h2.value, p_h2.value

        # Heat pump
        def heat_pump_optimal_value(problem_hp):
            """
            heat_pump_optimal_value obtains the optimal heat produced by the hydrogen system
            """
            # Solve the problem
            problem_hp.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_hp.value, l_hp.value

        # Heat consumer 1
        def consumer_1_optimal_value(problem_cons1):
            """
            consumer_1_optimal_value obtains the optimal heat consumed by consumer 1
            """
            # Solve the problem
            problem_cons1.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_cons1.value, l_cons1.value

        # Heat consumer 2
        def consumer_2_optimal_value(problem_cons2):
            """
            consumer_2_optimal_value obtains the optimal heat consumed by consumer 2
            """
            # Solve the problem
            problem_cons2.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_cons2.value, l_cons2.value

        # Heat consumer 3
        def consumer_3_optimal_value(problem_cons3):
            """
            consumer_3_optimal_value obtains the optimal heat consumed by consumer 3
            """
            # Solve the problem
            problem_cons3.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_cons3.value, l_cons3.value
        
        # Thermal energy storage
        def thermal_storage_optimal_value(problem_ts):
            """
            thermal_storage_optimal_value obtains the optimal heat stored in the thermal storage
            """
            # Solve the problem
            problem_ts.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_ts_net.value
        
        # Battery energy storage
        def battery_storage_optimal_value(problem_bat):
            """
            battery_storage_optimal_value obtains the optimal energy stored in the battery
            """
            # Solve the problem
            problem_bat.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return p_bat_net.value
        
        # Grid connection operation
        def grid_connection_optimal_value(problem_grid):
            """
            grid_connection_optimal_value obtains the optimal electricity import and export
            """
            # Solve the problem
            problem_grid.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return p_grid.value

        # -----------------------------------------
        # Calculating average values to be exchanged
        # -----------------------------------------
        def get_average_heat():
            """
            average_heat_value calculates the average heat value to be exchanged
            """
            return (q_cons1_fixed.value + q_cons2_fixed.value + q_cons3_fixed.value + q_ts_net_fixed.value \
                     - q_h2_fixed.value - q_hp_fixed.value) / n_heat_agents
        
        def get_average_elec():
            """
            average_elec_value calculates the average electricity value to be exchanged
            """
            return (l_cons1_fixed.value + l_cons2_fixed.value + l_cons3_fixed.value + p_h2_fixed.value + l_hp_fixed.value + p_bat_net_fixed.value + p_grid_fixed.value \
                    - p_pv.value) / n_elec_agents


        # Initial values
        lambda_heat_fixed.value       = np.zeros(T) # Lagrangian variable [CHF/kWh]
        lambda_elec_fixed.value       = np.zeros(T) # Lagrangian variable [CHF/kWh]
        p_grid_fixed.value            = np.zeros(T) # Initial value of the electricity grid power
        q_h2_fixed.value              = np.zeros(T) # Initial value of the hydrogen system heat
        p_h2_fixed.value              = np.zeros(T) # Initial value of the hydrogen system power
        q_hp_fixed.value              = np.zeros(T) # Initial value of the heat pump heat
        l_hp_fixed.value              = np.zeros(T) # Initial value of the heat pump load
        p_bat_net_fixed.value         = np.zeros(T) # Initial value of the battery net power
        q_ts_net_fixed.value          = np.zeros(T) # Initial value of the thermal storage input
        q_cons1_fixed.value           = np.zeros(T) # Initial value of the consumer 1 heat
        q_cons2_fixed.value           = np.zeros(T) # Initial value of the consumer 2 heat
        q_cons3_fixed.value           = np.zeros(T) # Initial value of the consumer 3 heat
        l_cons1_fixed.value           = np.zeros(T) # Initial value of the consumer 1 load
        l_cons2_fixed.value           = np.zeros(T) # Initial value of the consumer 2 load
        l_cons3_fixed.value           = np.zeros(T) # Initial value of the consumer 3 load

        q_average.value               = np.zeros(T) # Average heat value to be exchanged
        p_average.value               = np.zeros(T) # Average electricity value to be exchanged

        rho_heat.value                = rho_heat_fixed # Initial value of the penalty parameter
        rho_elec.value                = rho_elec_fixed # Initial value of the penalty parameter
        
        # Build the optimisation problems
        problem_h2      = hydrogen_system_optimisation()
        problem_hp      = heat_pump_optimisation()
        problem_cons1   = consumer_1_optimisation()
        problem_cons2   = consumer_2_optimisation()
        problem_cons3   = consumer_3_optimisation()
        problem_ts      = thermal_storage_optimisation()
        problem_bat     = battery_storage_optimisation()
        problem_grid    = grid_connection_optimisation()

        # Storing values over iterations - HEAT
        lambda_heat_storage         = np.zeros((T,max_iter))
        primal_gap_heat_storage     = np.zeros((T,max_iter))
        primal_norm_heat_storage    = np.zeros(max_iter)

        # Storing values over iterations - ELECTRICITY
        lambda_elec_storage         = np.zeros((T,max_iter))
        primal_gap_elec_storage     = np.zeros((T,max_iter))
        primal_norm_elec_storage    = np.zeros(max_iter)

        # Storing values over iterations - COST
        cost_storage        = np.zeros(max_iter)


        # ---------------------------------------------------------
        st = time.process_time()    # get the start time
        # ---------------------------------------------------------

        for iter in range(max_iter):

            # Update the hydrogen system
            q_h2_last, p_h2_last = hydrogen_system_optimal_value(problem_h2)
            q_h2_fixed.value     = q_h2_last
            p_h2_fixed.value     = p_h2_last

            q_average.value = get_average_heat()
            p_average.value = get_average_elec()

            # Update the heat pump
            q_hp_last, l_hp_last = heat_pump_optimal_value(problem_hp)
            q_hp_fixed.value     = q_hp_last
            l_hp_fixed.value     = l_hp_last

            q_average.value = get_average_heat()
            p_average.value = get_average_elec()

            # Update the heat consumer 1
            q_cons1_last, l_cons1_last      = consumer_1_optimal_value(problem_cons1)
            q_cons1_fixed.value             = q_cons1_last
            l_cons1_fixed.value             = l_cons1_last

            q_average.value = get_average_heat()
            p_average.value = get_average_elec()

            # Update the heat consumer 2
            q_cons2_last, l_cons2_last      = consumer_2_optimal_value(problem_cons2)
            q_cons2_fixed.value             = q_cons2_last
            l_cons2_fixed.value             = l_cons2_last

            q_average.value = get_average_heat()
            p_average.value = get_average_elec()

            # Update the heat consumer 3
            q_cons3_last, l_cons3_last      = consumer_3_optimal_value(problem_cons3)
            q_cons3_fixed.value             = q_cons3_last
            l_cons3_fixed.value             = l_cons3_last

            q_average.value = get_average_heat()
            p_average.value = get_average_elec()

            # Update the thermal storage - if selected
            if use_thermal_storage:
                q_ts_net_last               = thermal_storage_optimal_value(problem_ts)
                q_ts_net_fixed.value        = q_ts_net_last

                q_average.value = get_average_heat()
                p_average.value = get_average_elec()

            elif not use_thermal_storage and iter == 0:
                q_ts_net_last               = thermal_storage_optimal_value(problem_ts)


            # Update the battery storage - if selected
            if use_battery_storage:
                p_bat_net_last              = battery_storage_optimal_value(problem_bat)
                p_bat_net_fixed.value       = p_bat_net_last

                q_average.value = get_average_heat()
                p_average.value = get_average_elec()

            elif not use_battery_storage and iter == 0:
                p_bat_net_last              = battery_storage_optimal_value(problem_bat)

            # Update the grid connection
            p_grid_last            = grid_connection_optimal_value(problem_grid)
            p_grid_fixed.value   = p_grid_last

            q_average.value = get_average_heat()
            p_average.value = get_average_elec()

            # ---------------------------------------------------------
            # Updating calculations
            # ---------------------------------------------------------
            # Calculate the residuals
            primal_residual_heat    = get_average_heat() * n_heat_agents     # Difference between heat consumption and production at every timestep [kWh]
            primal_residual_elec    = get_average_elec() * n_elec_agents     # Difference between electricity consumption and production at every timestep [kWh]
            total_cost              = p_el_import.T @ p_imp.value - p_el_export.T @ p_exp.value + price_h2.T @ h2_imp.value - price_h2.T @ h2_exp.value # Total cost [CHF]

            # Storing values
            lambda_heat_storage[:,iter]          = lambda_heat_fixed.value
            primal_gap_heat_storage[:,iter]      = primal_residual_heat
            primal_norm_heat_storage[iter]       = np.linalg.norm(primal_residual_heat, ord = 1)

            lambda_elec_storage[:,iter]          = lambda_elec_fixed.value
            primal_gap_elec_storage[:,iter]      = primal_residual_elec
            primal_norm_elec_storage[iter]       = np.linalg.norm(primal_residual_elec, ord = 1)

            cost_storage[iter]                   = total_cost

            # ---------------------------------------------------------
            # Coordinator updates
            # ---------------------------------------------------------
            # Update the dual variable - heat value
            lambda_heat_fixed.value      = lambda_heat_fixed.value + rho_heat.value * (primal_residual_heat)
            dual_residual_heat           = lambda_heat_fixed.value - lambda_heat_storage[:,iter]
            # Update the dual variable - electricity value
            lambda_elec_fixed.value      = lambda_elec_fixed.value + rho_elec.value * (primal_residual_elec)
            dual_residual_elec           = lambda_elec_fixed.value - lambda_elec_storage[:,iter]

            # Norms of the residuals
            norm_primal_residual_heat   = np.linalg.norm(primal_residual_heat, ord = 1)
            norm_dual_residual_heat     = np.linalg.norm(dual_residual_heat, ord = 1)

            norm_primal_residual_elec   = np.linalg.norm(primal_residual_elec, ord = 1)
            norm_dual_residual_elec     = np.linalg.norm(dual_residual_elec, ord = 1)

            # Update penalty parameter
            # rho.value = 2 * rho.value

            # Check for convergence
            # if all(abs(primal_residual)) < tolerance_primal and all(abs(dual_residual)) < tolerance_dual:
            #     break

            print("Iteration: ",                                        iter," finished.\n")
            print("Current norm of primal residual (heat): ",           norm_primal_residual_heat," \n")
            print("Absolute change in dual this iteration: ",           norm_dual_residual_heat," \n")
            print("Current norm of primal residual (electricity): ",    norm_primal_residual_elec," \n")
            print("Absolute change in dual this iteration: ",           norm_dual_residual_elec," \n")
            print("Current total cost value: ",                         total_cost)

        # ---------------------------------------------------------
        et = time.process_time()    # get the end time
        res = et - st               # get execution time
        print('CPU Execution time:', res, 'seconds')
        # --------------------------------------------------------- 

        # --------------------------------------------------------------
        # Processing of results
        # --------------------------------------------------------------
        # ------ Main results -------
        total_cost              = total_cost                            # scalar 
        heat_value              = lambda_heat_fixed.value               # vector
        elec_value              = lambda_elec_fixed.value               # vector
        average_heat_value      = np.average(heat_value)                # scalar
        average_elec_value      = np.average(elec_value)                # scalar
        total_expenses          = p_el_import.T @ p_imp.value + price_h2.T @ h2_imp.value # scalar
        total_revenue           = p_el_export.T @ p_exp.value + price_h2.T @ h2_exp.value # scalar

        # ----- Cost results -------
        # Electricity
        electricity_expenses    = p_el_import.T @ p_imp.value # scalar
        electricity_revenue     = p_el_export.T @ p_exp.value # scalar
        # Hydrogen
        hydrogen_expenses       = price_h2.T @ h2_imp.value   # scalar
        hydrogen_revenue        = price_h2.T @ h2_exp.value   # scalar
        # H2 system
        expenses_h2_system      = elec_value.T @ l_h2.value + hydrogen_expenses  # scalar
        revenue_h2_system       = elec_value.T @ p_fc.value + hydrogen_revenue   # scalar 
        # Heat pump
        expenses_hp_system      = elec_value.T @ l_hp.value   # scalar 
        # PV system
        revenue_pv_system       = elec_value.T @ p_pv.value   # scalar
        # Consumers
        heat_cost_cons1         = heat_value.T @ q_cons1.value # scalar
        heat_cost_cons2         = heat_value.T @ q_cons2.value # scalar
        heat_cost_cons3         = heat_value.T @ q_cons3.value # scalar
        heat_cost_cons_total    = heat_cost_cons1 + heat_cost_cons2 + heat_cost_cons3 # scalar
        electricity_end_expenses= p_el_import.T @ (l_cons1.value + l_cons2.value + l_cons3.value) # scalar
        electricity_f_heat_expenses= elec_value.T @ (l_h2.value +l_hp.value) # scalar

        # ------ Heat results -------
        # General
        heat_consumption    = q_cons1.value + q_cons2.value + q_cons3.value + q_ts_in.value     # vector
        heat_demand         = q_cons1.value + q_cons2.value + q_cons3.value                     # vector
        heat_supply         = q_h2.value + q_hp.value + q_ts_out.value                          # vector
        heat_solar_gain     = q_gain * n_consumers                                              # vector
        # Hydrogen
        heat_h2_used        = q_h2.value        # vector
        heat_el_used        = q_el.value        # vector
        heat_fc_used        = q_fc.value        # vector
        heat_el_gen         = q_gen_el.value    # vector
        heat_fc_gen         = q_gen_fc.value    # vector
        heat_waste          = q_was_el.value + q_was_fc.value                                   # vector
        heat_waste_el       = q_was_el.value                                                    # vector
        heat_waste_fc       = q_was_fc.value                                                    # vector
        # Heat pump
        heat_hp_used        = q_hp.value        # vector
        # Thermal storage
        heat_ts_used        = q_ts_out.value    # vector
        heat_ts_input       = q_ts_in.value     # vector
        energy_ts_stored    = e_ts_sto.value    # vector
        total_heat_consumption= np.sum(heat_consumption)  # scalar
        total_heat_demand   = np.sum(heat_demand)       # scalar
        total_heat_supply   = np.sum(heat_supply)       # scalar
        total_heat_solar_gain= np.sum(heat_solar_gain)  # scalar
        total_heat_waste    = np.sum(heat_waste)        # scalar
        total_heat_waste_el = np.sum(heat_waste_el)     # scalar
        total_heat_waste_fc = np.sum(heat_waste_fc)     # scalar
        total_heat_h2_used  = np.sum(heat_h2_used)      # scalar
        total_heat_el_used  = np.sum(heat_el_used)      # scalar
        total_heat_fc_used  = np.sum(heat_fc_used)      # scalar
        total_heat_el_gen   = np.sum(heat_el_gen)       # scalar
        total_heat_fc_gen   = np.sum(heat_fc_gen)       # scalar
        total_heat_hp_used  = np.sum(heat_hp_used)      # scalar
        total_heat_ts_used  = np.sum(heat_ts_used)      # scalar
        total_heat_ts_input = np.sum(heat_ts_input)     # scalar
        total_energy_ts_stored = np.sum(energy_ts_stored) # scalar

        # ------ Electricity results -------
        # General
        electricity_for_heat_demand = l_h2.value + l_hp.value        # vector
        electricity_end_demand      = l_cons1.value + l_cons2.value + l_cons3.value    # vector
        electricity_total_demand    = electricity_for_heat_demand + electricity_end_demand # vector
        electricity_generated       = p_fc.value + p_pv.value   # vector # TODO: could add here Battery
        electricity_imported        = p_imp.value               # vector
        electricity_exported        = p_exp.value               # vector
        electricity_net_imported    = electricity_imported - electricity_exported # vector
        # Hydrogen
        electricity_h2_demand       = l_h2.value                # vector
        electricity_el_demand       = l_el.value                # vector
        electricity_co_demand       = l_co.value                # vector
        electricity_fc_produced     = p_fc.value                # vector
        # Heat pump 
        electricity_hp_demand       = l_hp.value                # vector
        # PV
        electricity_pv_produced     = p_pv.value                # vector
        # Battery storage
        electricity_bat_charged     = p_bat_ch.value            # vector
        electricity_bat_discharged  = p_bat_dis.value           # vector
        energy_bat_stored           = e_bat.value               # vector
        # Totals
        total_electricity_for_heat_demand   = np.sum(electricity_for_heat_demand)       # scalar
        total_electricity_end_demand        = np.sum(electricity_end_demand)            # scalar
        total_electricity_total_demand      = np.sum(electricity_total_demand)          # scalar
        total_electricity_generated         = np.sum(electricity_generated)             # scalar
        total_electricity_imported          = np.sum(electricity_imported)              # scalar
        total_electricity_exported          = np.sum(electricity_exported)              # scalar
        total_electricity_h2_demand         = np.sum(electricity_h2_demand)             # scalar
        total_electricity_el_demand         = np.sum(electricity_el_demand)             # scalar
        total_electricity_co_demand         = np.sum(electricity_co_demand)             # scalar
        total_electricity_hp_demand         = np.sum(electricity_hp_demand)             # scalar
        total_electricity_fc_produced       = np.sum(electricity_fc_produced)           # scalar
        total_electricity_pv_produced       = np.sum(electricity_pv_produced)           # scalar
        total_electricity_bat_charged       = np.sum(electricity_bat_charged)           # scalar
        total_electricity_bat_discharged    = np.sum(electricity_bat_discharged)        # scalar
        total_energy_bat_stored             = np.sum(energy_bat_stored)                 # scalar
        total_net_electricity_imported      = total_electricity_imported - total_electricity_exported # scalar

        # ------ Hydrogen results -------
        hydrogen_consumption    = h2_fc.value   # vector
        hydrogen_production     = h2_prod.value # vector
        hydrogen_imported       = h2_imp.value  # vector
        hydrogen_exported       = h2_exp.value  # vector
        hydrogen_storage_level  = h2_sto.value  # vector
        total_hydrogen_used     = np.sum(hydrogen_consumption) # scalar
        total_hydrogen_produced = np.sum(hydrogen_production)  # scalar
        total_hydrogen_imported = np.sum(hydrogen_imported)    # scalar
        total_hydrogen_exported = np.sum(hydrogen_exported)    # scalar
        avg_hydrogen_storage    = np.average(hydrogen_storage_level) # scalar

        # ------ Temperature results -------
        temp_cons1              = T_cons1.value # vector
        temp_cons2              = T_cons2.value # vector
        temp_cons3              = T_cons3.value # vector
        avg_temp_cons1          = np.average(temp_cons1) # scalar
        avg_temp_cons2          = np.average(temp_cons2) # scalar
        avg_temp_cons3          = np.average(temp_cons3) # scalar
        max_temp_cons1          = np.max(temp_cons1) # scalar
        max_temp_cons2          = np.max(temp_cons2) # scalar
        max_temp_cons3          = np.max(temp_cons3) # scalar

        # -------- Rate of change of variables --------
        # Electricity
        rate_elec_f_heat_demand     = np.diff(electricity_for_heat_demand, prepend=electricity_for_heat_demand[0])  # vector
        rate_elec_end_demand        = np.diff(electricity_end_demand, prepend=electricity_end_demand[0])            # vector
        rate_elec_total_demand      = np.diff(electricity_total_demand, prepend=electricity_total_demand[0])        # vector
        rate_elec_generated         = np.diff(electricity_generated, prepend=electricity_generated[0]) 
        rate_elec_imported          = np.diff(electricity_imported, prepend=electricity_imported[0])
        rate_elec_exported          = np.diff(electricity_exported, prepend=electricity_exported[0])
        # Heat
        rate_heat_consumption       = np.diff(heat_consumption, prepend=heat_consumption[0]) # vector
        rate_heat_demand            = np.diff(heat_demand, prepend=heat_demand[0])          # vector
        rate_heat_supply            = np.diff(heat_supply, prepend=heat_supply[0])          # vector
        # Temperature
        rate_temp_cons1             = np.diff(temp_cons1, prepend=temp_cons1[0])      # vector
        rate_temp_cons2             = np.diff(temp_cons2, prepend=temp_cons2[0])      # vector
        rate_temp_cons3             = np.diff(temp_cons3, prepend=temp_cons3[0])      # vector
        # -------- Average rates of absolute values
        avg_rate_elec_f_heat_demand = np.average(abs(rate_elec_f_heat_demand))  # scalar
        avg_rate_elec_end_demand    = np.average(abs(rate_elec_end_demand))     # scalar
        avg_rate_elec_total_demand  = np.average(abs(rate_elec_total_demand))   # scalar
        avg_rate_elec_generated     = np.average(abs(rate_elec_generated))      # scalar
        avg_rate_elec_imported      = np.average(abs(rate_elec_imported))       # scalar
        avg_rate_elec_exported      = np.average(abs(rate_elec_exported))       # scalar
        avg_rate_heat_demand        = np.average(abs(rate_heat_demand))         # scalar
        avg_rate_heat_supply        = np.average(abs(rate_heat_supply))         # scalar
        avg_rate_temp_cons1         = np.average(abs(rate_temp_cons1))          # scalar
        avg_rate_temp_cons2         = np.average(abs(rate_temp_cons2))          # scalar
        avg_rate_temp_cons3         = np.average(abs(rate_temp_cons3))          # scalar
        # -------- Average rates of only ramp up values
        avg_rate_elec_f_heat_demand_ramp_up = np.average(rate_elec_f_heat_demand[rate_elec_f_heat_demand > 0])  # scalar
        avg_rate_elec_end_demand_ramp_up    = np.average(rate_elec_end_demand[rate_elec_end_demand > 0])        # scalar
        avg_rate_elec_total_demand_ramp_up  = np.average(rate_elec_total_demand[rate_elec_total_demand > 0])    # scalar
        avg_rate_elec_generated_ramp_up     = np.average(rate_elec_generated[rate_elec_generated > 0])          # scalar
        avg_rate_elec_imported_ramp_up      = np.average(rate_elec_imported[rate_elec_imported > 0])            # scalar
        avg_rate_elec_exported_ramp_up      = np.average(rate_elec_exported[rate_elec_exported > 0])            # scalar
        avg_rate_heat_demand_ramp_up        = np.average(rate_heat_demand[rate_heat_demand > 0])                # scalar
        avg_rate_heat_supply_ramp_up        = np.average(rate_heat_supply[rate_heat_supply > 0])                # scalar
        avg_rate_temp_cons1_ramp_up         = np.average(rate_temp_cons1[rate_temp_cons1 > 0])                  # scalar
        avg_rate_temp_cons2_ramp_up         = np.average(rate_temp_cons2[rate_temp_cons2 > 0])                  # scalar
        avg_rate_temp_cons3_ramp_up         = np.average(rate_temp_cons3[rate_temp_cons3 > 0])                  # scalar
        # -------- Maximum rates
        max_rate_elec_f_heat_demand = np.max(abs(rate_elec_f_heat_demand))  # scalar
        max_rate_elec_end_demand    = np.max(abs(rate_elec_end_demand))     # scalar
        max_rate_elec_total_demand  = np.max(abs(rate_elec_total_demand))   # scalar
        max_rate_elec_generated     = np.max(abs(rate_elec_generated))      # scalar
        max_rate_elec_imported      = np.max(abs(rate_elec_imported))       # scalar
        max_rate_elec_exported      = np.max(abs(rate_elec_exported))       # scalar
        max_rate_heat_demand        = np.max(abs(rate_heat_demand))         # scalar
        max_rate_heat_supply        = np.max(abs(rate_heat_supply))         # scalar
        max_rate_temp_cons1         = np.max(abs(rate_temp_cons1))          # scalar
        max_rate_temp_cons2         = np.max(abs(rate_temp_cons2))          # scalar
        max_rate_temp_cons3         = np.max(abs(rate_temp_cons3))          # scalar
        
        # Distributed results
        absolute_deviation_heat     = norm_primal_residual_heat # scalar
        absolute_deviation_elec     = norm_primal_residual_elec # scalar

        # ------- Print main results -------
        print("Optimal system cost: ",          total_cost)
        print("Total expenses: ",               total_expenses)
        print("Total revenue: ",                total_revenue)
        print("Average heat value: ",           average_heat_value)
        print("Average electricity value: ",    average_elec_value)
        print("Total heat demand:" ,            total_heat_demand)
        print("Total electricity demanded: ",   total_electricity_total_demand)
        print("Total electricity produced: ",   total_electricity_generated)

        if save_results:
            # Make sure the results folder exists
            rhoname_heat = str(rho_heat.value).replace(".","_")
            rhoname_elec = str(rho_elec.value).replace(".","_")
            rhoname = rhoname_heat + "U" + rhoname_elec
            folder_path = f"results/{name_version}/rho_{rhoname}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            else:
                print(folder_path, " folder already exists.")

            # ---------------------------------------------------------
            # Save computation time in a text file
            with open(f'{folder_path}/computation_time.txt', 'w') as f:
                f.write(f'CPU Execution time: {res} seconds')
            # ---------------------------------------------------------

            # Results over iterations
            results_iterations = pd.DataFrame({"Iteration": range(iter),
                                            "PrimalGap": primal_norm_heat_storage[0:iter],
                                            "PrimalGapElec": primal_norm_elec_storage[0:iter],
                                            "Cost": cost_storage[0:iter],
                                            "HeatValue": np.sum(lambda_heat_storage, axis=0)[0:iter],
                                            "ElecValue": np.sum(lambda_elec_storage, axis=0)[0:iter]})
            # Save to csv
            results_iterations.to_csv(f"{folder_path}/convergence.csv", index=False)
            
            # Saving the scalar results in a csv file - columns: name, value, unit
            scalar_results = {
                'total_cost':                       (total_cost, 'CHF'),
                'average_heat_value':               (average_heat_value, 'CHF/kWh'),
                'average_elec_value':               (average_elec_value, 'CHF/kWh'),
                'total_expenses':                   (total_expenses, 'CHF'),
                'total_revenue':                    (total_revenue, 'CHF'),
                'electricity_expenses':             (electricity_expenses, 'CHF'),
                'electricity_revenue':              (electricity_revenue, 'CHF'),
                'hydrogen_expenses':                (hydrogen_expenses, 'CHF'),
                'hydrogen_revenue':                 (hydrogen_revenue, 'CHF'),
                'expenses_h2_system':               (expenses_h2_system, 'CHF'),
                'revenue_h2_system':                (revenue_h2_system, 'CHF'),
                'expenses_hp_system':               (expenses_hp_system, 'CHF'),
                'revenue_pv_system':                (revenue_pv_system, 'CHF'),
                'heat_cost_cons1':                  (heat_cost_cons1, 'CHF'),
                'heat_cost_cons2':                  (heat_cost_cons2, 'CHF'),
                'heat_cost_cons3':                  (heat_cost_cons3, 'CHF'),
                'heat_cost_cons_total':             (heat_cost_cons_total, 'CHF'),
                'electricity_end_expenses':         (electricity_end_expenses, 'CHF'),
                'electricity_f_heat_expenses':      (electricity_f_heat_expenses, 'CHF'),
                'total_heat_consumption':           (total_heat_consumption, 'kWh'),
                'total_heat_demand':                (total_heat_demand, 'kWh'),
                'total_heat_supply':                (total_heat_supply, 'kWh'),
                'total_heat_solar_gain':            (total_heat_solar_gain, 'kWh'),
                'total_heat_waste':                 (total_heat_waste, 'kWh'),
                'total_heat_waste_el':              (total_heat_waste_el, 'kWh'),
                'total_heat_waste_fc':              (total_heat_waste_fc, 'kWh'),
                'total_heat_h2_used':               (total_heat_h2_used, 'kWh'),
                'total_heat_el_used':               (total_heat_el_used, 'kWh'),
                'total_heat_fc_used':               (total_heat_fc_used, 'kWh'),
                'total_heat_el_gen':                (total_heat_el_gen, 'kWh'),
                'total_heat_fc_gen':                (total_heat_fc_gen, 'kWh'),
                'total_heat_hp_used':               (total_heat_hp_used, 'kWh'),
                'total_heat_ts_used':               (total_heat_ts_used, 'kWh'),
                'total_heat_ts_input':              (total_heat_ts_input, 'kWh'),
                'total_energy_ts_stored':           (total_energy_ts_stored, 'kWh'),
                'total_electricity_for_heat_demand':(total_electricity_for_heat_demand, 'kWh'),
                'total_electricity_end_demand':     (total_electricity_end_demand, 'kWh'),
                'total_electricity_total_demand':   (total_electricity_total_demand, 'kWh'),
                'total_electricity_generated':      (total_electricity_generated, 'kWh'),
                'total_electricity_imported':       (total_electricity_imported, 'kWh'),
                'total_electricity_exported':       (total_electricity_exported, 'kWh'),
                'total_electricity_h2_demand':      (total_electricity_h2_demand, 'kWh'),
                'total_electricity_el_demand':      (total_electricity_el_demand, 'kWh'),
                'total_electricity_co_demand':      (total_electricity_co_demand, 'kWh'),
                'total_electricity_hp_demand':      (total_electricity_hp_demand, 'kWh'),
                'total_electricity_fc_produced':    (total_electricity_fc_produced, 'kWh'),
                'total_electricity_pv_produced':    (total_electricity_pv_produced, 'kWh'),
                'total_electricity_bat_charged':    (total_electricity_bat_charged, 'kWh'),
                'total_electricity_bat_discharged': (total_electricity_bat_discharged, 'kWh'),
                'total_energy_bat_stored':          (total_energy_bat_stored, 'kWh'),
                'total_net_electricity_imported':   (total_net_electricity_imported, 'kWh'),
                'total_hydrogen_used':              (total_hydrogen_used, 'kg'),
                'total_hydrogen_produced':          (total_hydrogen_produced, 'kg'),
                'total_hydrogen_imported':          (total_hydrogen_imported, 'kg'),
                'total_hydrogen_exported':          (total_hydrogen_exported, 'kg'),
                'avg_hydrogen_storage':             (avg_hydrogen_storage, 'kg'),
                'avg_temp_cons1':                   (avg_temp_cons1, '°C'),
                'avg_temp_cons2':                   (avg_temp_cons2, '°C'),
                'avg_temp_cons3':                   (avg_temp_cons3, '°C'),
                'max_temp_cons1':                   (max_temp_cons1, '°C'),
                'max_temp_cons2':                   (max_temp_cons2, '°C'),
                'max_temp_cons3':                   (max_temp_cons3, '°C'),
                'avg_rate_elec_f_heat_demand':      (avg_rate_elec_f_heat_demand, 'kWh/h'),
                'avg_rate_elec_end_demand':         (avg_rate_elec_end_demand, 'kWh/h'),
                'avg_rate_elec_total_demand':       (avg_rate_elec_total_demand, 'kWh/h'),
                'avg_rate_elec_generated':          (avg_rate_elec_generated, 'kWh/h'),
                'avg_rate_elec_imported':           (avg_rate_elec_imported, 'kWh/h'),
                'avg_rate_elec_exported':           (avg_rate_elec_exported, 'kWh/h'),
                'avg_rate_heat_demand':             (avg_rate_heat_demand, 'kWh/h'),
                'avg_rate_heat_supply':             (avg_rate_heat_supply, 'kWh/h'),
                'avg_rate_temp_cons1':              (avg_rate_temp_cons1, '°C/h'),
                'avg_rate_temp_cons2':              (avg_rate_temp_cons2, '°C/h'),
                'avg_rate_temp_cons3':              (avg_rate_temp_cons3, '°C/h'),
                'avg_rate_elec_f_heat_demand_ramp_up':(avg_rate_elec_f_heat_demand_ramp_up, 'kWh/h'),
                'avg_rate_elec_end_demand_ramp_up': (avg_rate_elec_end_demand_ramp_up, 'kWh/h'),
                'avg_rate_elec_total_demand_ramp_up':(avg_rate_elec_total_demand_ramp_up, 'kWh/h'),
                'avg_rate_elec_generated_ramp_up':  (avg_rate_elec_generated_ramp_up, 'kWh/h'),
                'avg_rate_elec_imported_ramp_up':   (avg_rate_elec_imported_ramp_up, 'kWh/h'),
                'avg_rate_elec_exported_ramp_up':   (avg_rate_elec_exported_ramp_up, 'kWh/h'),
                'avg_rate_heat_demand_ramp_up':     (avg_rate_heat_demand_ramp_up, 'kWh/h'),
                'avg_rate_heat_supply_ramp_up':     (avg_rate_heat_supply_ramp_up, 'kWh/h'),
                'avg_rate_temp_cons1_ramp_up':      (avg_rate_temp_cons1_ramp_up, '°C/h'),
                'avg_rate_temp_cons2_ramp_up':      (avg_rate_temp_cons2_ramp_up, '°C/h'),
                'avg_rate_temp_cons3_ramp_up':      (avg_rate_temp_cons3_ramp_up, '°C/h'),
                'max_rate_elec_f_heat_demand':      (max_rate_elec_f_heat_demand, 'kWh/h'),
                'max_rate_elec_end_demand':         (max_rate_elec_end_demand, 'kWh/h'),
                'max_rate_elec_total_demand':       (max_rate_elec_total_demand, 'kWh/h'),
                'max_rate_elec_generated':          (max_rate_elec_generated, 'kWh/h'),
                'max_rate_elec_imported':           (max_rate_elec_imported, 'kWh/h'),
                'max_rate_elec_exported':           (max_rate_elec_exported, 'kWh/h'),
                'max_rate_heat_demand':             (max_rate_heat_demand, 'kWh/h'),
                'max_rate_heat_supply':             (max_rate_heat_supply, 'kWh/h'),
                'max_rate_temp_cons1':              (max_rate_temp_cons1, '°C/h'),
                'max_rate_temp_cons2':              (max_rate_temp_cons2, '°C/h'),
                'max_rate_temp_cons3':              (max_rate_temp_cons3, '°C/h'),
            }
            # Convert to DataFrame
            scalar_results_df = pd.DataFrame([(key, value[0], value[1]) for key, value in scalar_results.items()], columns=['Metric', 'Value', 'Unit'])
            # Save to CSV
            scalar_results_csv = f"{folder_path}/scalar_results.csv"
            scalar_results_df.to_csv(scalar_results_csv, index=False)

            # Main results
            time_index = pd.date_range(start='2023-01-01 00:00:00', end='2023-12-31 23:00:00', freq='H')
            results = pd.DataFrame({"Time": np.arange(0,T,1),
                                    "DateTime":                     time_index,
                                    "PriceImportElectricity":       p_el_import,
                                    "PriceExportElectricity":       p_el_export,
                                    "PriceHydrogen":                price_h2,
                                    "TemperatureAmbient":           T_amb,
                                    "HeatGain":                     heat_solar_gain,
                                    "ElecEndDemand":                electricity_end_demand,
                                    "HeatValue":                    heat_value,
                                    "ElecValue":                    elec_value,
                                    "HeatConsumption":              heat_consumption,
                                    "HeatDemand":                   heat_demand,
                                    "HeatSupply":                   heat_supply,
                                    "HeatDemand1":                  q_cons1.value,
                                    "HeatDemand2":                  q_cons2.value,
                                    "HeatDemand3":                  q_cons3.value,
                                    "HeatH2":                       heat_h2_used,
                                    "HeatWaste":                    heat_waste,
                                    "HeatWasteEl":                  heat_waste_el,
                                    "HeatWasteFC":                  heat_waste_fc,
                                    "HeatH2":                       heat_h2_used,     
                                    "HeatEl":                       heat_el_used,
                                    "HeatFC":                       heat_fc_used,
                                    "HeatElGen":                    heat_el_gen,
                                    "HeatFCGen":                    heat_fc_gen,
                                    "HeatHP":                       heat_hp_used,
                                    "HeatTS":                       heat_ts_used,
                                    "HeatTSIn":                     heat_ts_input,
                                    "EnergyTSStored":               energy_ts_stored,
                                    "HeatAvgDemand":                heat_demand/n_consumers,
                                    "ElectricityForHeatDemand":     electricity_for_heat_demand,
                                    "ElectricityEndDemand":         electricity_end_demand,
                                    "ElectricityTotalDemand":       electricity_total_demand,
                                    "ElectricityGenerated":         electricity_generated,
                                    "ElectricityImported":          electricity_imported,
                                    "ElectricityExported":          electricity_exported,
                                    "NetElectricityImported":       electricity_net_imported,
                                    "LoadH2System":                 electricity_h2_demand,
                                    "LoadElectrolyser":             electricity_el_demand,
                                    "LoadCompressor":               electricity_co_demand,
                                    "LoadHeatPump":                 electricity_hp_demand,
                                    "ElectricityFCProduced":        electricity_fc_produced,
                                    "ElectricityPVProduced":        electricity_pv_produced,
                                    "ElectricityBatteryCharged":    electricity_bat_charged,
                                    "ElectricityBatteryDischarged": electricity_bat_discharged,
                                    "EnergyBatteryStored":          energy_bat_stored,
                                    "H2Consumed":                   hydrogen_consumption,
                                    "H2Produced":                   hydrogen_production,
                                    "H2FCUsed":                     hydrogen_consumption,
                                    "H2Stored":                     hydrogen_storage_level,
                                    "H2Imported":                   hydrogen_imported,
                                    "H2Exported":                   hydrogen_exported,
                                    "TCons1":                       temp_cons1,
                                    "TCons2":                       temp_cons2,
                                    "TCons3":                       temp_cons3,
                                    "TConsAvg":                     (temp_cons1 + temp_cons2 + temp_cons3) / n_consumers,
                                    "TCons1Slack":                  T_cons1_slack.value,
                                    "TCons2Slack":                  T_cons2_slack.value,
                                    "TCons3Slack":                  T_cons3_slack.value,
                                    "RateElecFHeatDemand":          rate_elec_f_heat_demand,
                                    "RateElecEndDemand":            rate_elec_end_demand,
                                    "RateElecTotalDemand":          rate_elec_total_demand,
                                    "RateElecGenerated":            rate_elec_generated,
                                    "RateElecImported":             rate_elec_imported,
                                    "RateElecExported":             rate_elec_exported,
                                    "RateHeatConsumption":          rate_heat_consumption,
                                    "RateHeatDemand":               rate_heat_demand,
                                    "RateHeatSupply":               rate_heat_supply,
                                    "RateTempCons1":                rate_temp_cons1,
                                    "RateTempCons2":                rate_temp_cons2,
                                    "RateTempCons3":                rate_temp_cons3, 
                                    })
            # Add time_index as a column
            results["DateTime"] = time_index
            # Save to csv
            results.to_csv(f"{folder_path}/time_data_{name_version}.csv", index=False)

            # Plot the results
            
            # -------------------------------------------------------------------
            # ----- Plot the convergence results (distributed version only) -----
            # -------------------------------------------------------------------

            # Cost convergence
            fig, ax = plt.subplots()
            ax.plot(cost_storage[0:iter], label="Cost [CHF]")
            ax.axhline(y = min_cost_centralised, color = 'r', linestyle = 'dotted', label = "Centralised cost")
            ax.set(xlabel="Iteration", 
                ylabel="Cost [CHF]", 
                title="Convergence to Minimum Cost")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/convergence_cost_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Heat balance convergence
            fig, ax = plt.subplots()
            ax.plot(primal_norm_heat_storage[0:iter], label="Primal Gap [kWh]")
            ax.set(xlabel="Iteration",
                ylabel="Norm of residual [kWh]",
                title="Convergence to feasible solution")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/convergence_residual_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Electricity balance convergence
            fig, ax = plt.subplots()
            ax.plot(primal_norm_elec_storage[0:iter], label="Primal Gap [kWh]")
            ax.set(xlabel="Iteration",
                ylabel="Norm of residual [kWh]",
                title="Convergence to feasible solution")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/convergence_residual_elec_{name_version}.pdf', format="pdf", bbox_inches="tight")


            # -----------------------------------------------------------
            #  Plotting the time series results
            # -----------------------------------------------------------

            #### Yearly results ####

            # Electricity results ---------------------
            # Plot the electricity value
            fig, ax = plt.subplots()
            ax.plot(time_index, elec_value, label="Electricity Value", color='dodgerblue', linestyle='dashed')
            ax.set_xlabel("Date")
            ax.set_ylabel("Value [CHF/kWh]")
            ax.set_title("Electricity Value")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/electricity_value_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the electricity total demand
            fig, ax = plt.subplots()
            ax.plot(time_index, electricity_for_heat_demand, label="Electricity for heat", color='indianred')
            ax.plot(time_index, electricity_end_demand, label="Electricity end demand", color='goldenrod')
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Electricity Demand")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/electricity_demand_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Stacked plot of electricity demand
            fig, ax = plt.subplots()
            ax.stackplot(time_index, electricity_for_heat_demand, electricity_end_demand, labels=["Electricity for heat", "Electricity end demand"], colors=['indianred', 'goldenrod'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Electricity Demand")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/electricity_demand_stacked_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot cumulative electricity demand
            cum_electricity_for_heat_demand = np.cumsum(electricity_for_heat_demand)
            cum_electricity_end_demand = np.cumsum(electricity_end_demand)
            fig, ax = plt.subplots()
            ax.stackplot(time_index, cum_electricity_for_heat_demand, cum_electricity_end_demand, labels=["Electricity for heat", "Electricity end demand"], colors=['indianred', 'goldenrod'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kWh]")
            ax.set_title("Cumulative Electricity Demand")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/cumulative_electricity_demand_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the electricity generation
            fig, ax = plt.subplots()
            ax.plot(time_index, electricity_generated, label="Electricity Generation", color='limegreen')
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Electricity Generation")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/electricity_generation_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot stacked electricity generation
            fig, ax = plt.subplots()
            ax.stackplot(time_index, electricity_pv_produced, electricity_fc_produced, labels=["PV", "Fuel Cell"], colors=['gold', 'orangered'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Electricity Generation")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/electricity_generation_stacked_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot cumulative electricity generation from sources in stacked plot
            cum_electricity_pv_produced = np.cumsum(electricity_pv_produced)
            cum_electricity_fc_produced = np.cumsum(electricity_fc_produced)
            fig, ax = plt.subplots()
            ax.stackplot(time_index, cum_electricity_pv_produced, cum_electricity_fc_produced, labels=["PV", "Fuel Cell"], colors=['gold', 'orangered'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kWh]")
            ax.set_title("Cumulative Electricity Generation")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/cumulative_electricity_generation_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot net electricity imports
            fig, ax = plt.subplots()
            ax.plot(time_index, electricity_net_imported, label="Net electricity [kW]") 
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Net Electricity")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/net_electricity_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot electricity import
            fig, ax = plt.subplots()
            ax.plot(time_index, electricity_imported, label="Import electricity [kW]", color='slategrey')
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Electricity Import")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/electricity_import_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot electricity export
            fig, ax = plt.subplots()
            ax.plot(time_index, electricity_exported, label="Export electricity [kW]", color='peru')
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Electricity Export")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/electricity_export_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plotting import and export as area plots, net electricity as line plot. Export is negative.
            fig, ax = plt.subplots()
            ax.fill_between(time_index, electricity_imported, 0, label=r"Import, $P_\mathrm{imp}$", color='slategrey', alpha=0.5)
            ax.fill_between(time_index, - electricity_exported, 0, label=r"Export, $P_\mathrm{exp}$", color='peru', alpha=0.5)
            ax.plot(time_index, electricity_net_imported, label="Net imports", color='black', alpha = 1, linestyle = 'dashed')
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Electricity Demand and Supply")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/electricity_net_import_export_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the electricity price
            fig, ax = plt.subplots()
            ax.plot(time_index, p_el_import, label=r"Import, $p^{E}_\mathrm{imp}$", color='slategrey')
            ax.plot(time_index, p_el_export, label=r"Export, $p^{E}_\mathrm{exp}$", color='peru')
            ax.set_xlabel("Date")
            ax.set_ylabel("Electricity Price [CHF/kWh]")
            ax.set_title("Electricity Price")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/electricity_price_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            
            # Heat results ---------------------
            # Plot the heat demand
            fig, ax = plt.subplots()
            ax.plot(time_index, q_cons1.value, label="Consumer 1", color='chocolate', linestyle='dotted')
            ax.plot(time_index, q_cons2.value, label="Consumer 2", color='slategrey', linestyle='dotted')
            ax.plot(time_index, q_cons3.value, label="Consumer 3", color='darkseagreen', linestyle='dotted')
            ax.plot(time_index, heat_demand, label="Total", color='black', linestyle='solid')
            ax.set_xlabel("Date")
            ax.set_ylabel("Heat [kW]")
            ax.set_title("Heat Demand")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/heat_demand_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the heat supply
            fig, ax = plt.subplots()
            ax.plot(time_index, heat_h2_used, label="Hydrogen heat", color='limegreen')
            ax.plot(time_index, heat_hp_used, label="Heat Pump", color='darkkhaki')
            ax.plot(time_index, heat_supply, label="Total", color='black', linestyle='solid')
            ax.set_xlabel("Date")
            ax.set_ylabel("Heat [kW]")
            ax.set_title("Heat Supply")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/heat_supply_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot stacked heat supply
            fig, ax = plt.subplots()
            ax.stackplot(time_index, heat_el_used, heat_fc_used, heat_hp_used, labels=["Electrolyser", "Fuel Cell", "Heat Pump"], colors=[ 'royalblue', 'orangered','darkkhaki'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Heat [kW]")
            ax.set_title("Heat Supply")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/heat_supply_stacked_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the cumulative heat generation from sources in stacked plot
            cum_heat_el = np.cumsum(heat_el_used)
            cum_heat_fc = np.cumsum(heat_fc_used)
            cum_heat_hp = np.cumsum(heat_hp_used)
            fig, ax = plt.subplots()
            ax.stackplot(time_index, cum_heat_el, cum_heat_fc, cum_heat_hp, labels=["Electrolyser", "Fuel Cell", "Heat Pump"], colors=['royalblue', 'orangered', 'darkkhaki'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Heat Generation [kWh]")
            ax.set_title("Heat Generation")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/heat_generation_stacked_cumulative_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the heat value
            fig, ax = plt.subplots()
            ax.plot(time_index, heat_value, label="Heat value", color = 'red', linestyle='dashed')
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Heat Value [CHF/kWh]")
            ax.set_title("Heat Value")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/heat_value_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the total heat demand and heat value on two separate y axes
            fig, ax1 = plt.subplots()
            ax1.plot(time_index, heat_demand, label="Heat Demand", color='black', marker='o')
            ax.set_xlabel("Date")
            ax1.set_ylabel("Heat Demand [kW]")
            ax1.set_title("Heat Demand and Value")
            ax2 = ax1.twinx()
            ax2.plot(time_index, heat_value, label="Heat Value", color='red', linestyle='dashed', marker='x')
            ax2.set_ylabel("Heat Value [CHF/kWh]")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/heat_demand_value_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plotting all heat wasted and heat dumped slack in cumulative plot
            cum_heat_waste_el = np.cumsum(heat_waste_el)
            cum_heat_waste_fc = np.cumsum(heat_waste_fc)
            fig, ax = plt.subplots()
            ax.stackplot(time_index, cum_heat_waste_el, cum_heat_waste_fc, labels=["Waste Electrolyser", "Waste Fuel Cell"], colors=['royalblue', 'orangered'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Heat [kWh]")
            ax.set_title("Cumulative Heat Wasted")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/heat_wasted_stacked_cumulative_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Hydrogen results ---------------------
            # Plot the hydrogen storage
            fig, ax = plt.subplots()
            ax.plot(time_index, hydrogen_storage_level, label="Hydrogen Storage [kg]", color='cadetblue')
            ax.set_xlabel("Date")
            ax.set_ylabel("Storage [kg]")
            ax.set_title("Hydrogen Storage")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/hydrogen_storage_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the hydrogen production, consumption, import and export
            fig, ax = plt.subplots()
            ax.plot(time_index, hydrogen_production, label="Production", color = 'teal')
            ax.plot(time_index, hydrogen_consumption, label="Consumption", color = 'darkorange')
            ax.plot(time_index, hydrogen_imported, label="Import", linestyle='dotted', color='slategrey')
            ax.plot(time_index, hydrogen_exported, label="Export", linestyle='dotted', color = 'peru')
            ax.set_xlabel("Date")
            ax.set_ylabel("Hydrogen [kg]")
            ax.set_title("Hydrogen Production, Consumption, Import and Export")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/hydrogen_four_options_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the hydrogen in, out and storage level
            fig, ax = plt.subplots()
            ax.plot(time_index, hydrogen_imported + hydrogen_production, label="In", color='slategrey', linestyle='dotted')
            ax.plot(time_index, hydrogen_exported + hydrogen_consumption, label="Out", color='peru', linestyle='dotted')
            # Plot storage level colored in
            ax.fill_between(time_index, hydrogen_storage_level, color='cadetblue', alpha=0.5, label="Storage")
            ax.set_xlabel("Date")
            ax.set_ylabel("Hydrogen [kg]")
            ax.set_title("Hydrogen In, Out and Storage Level")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/hydrogen_in_out_storage_year_{name_version}.pdf', format="pdf", bbox_inches = "tight")

            # Temperature results ---------------------
            # Plot the ambient temperature
            fig, ax = plt.subplots()
            ax.plot(time_index, T_amb, label="Ambient", color='dimgrey', linestyle='dotted')
            ax.set_xlabel("Date")
            ax.set_ylabel("Temperature [°C]")
            ax.set_title("Ambient Temperature")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/ambient_temperature_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the consumer temperature
            avg_cons_temperature = (temp_cons1 + temp_cons2 + temp_cons3) / n_consumers
            fig, ax = plt.subplots()
            ax.plot(time_index, temp_cons1, label="Consumer 1", linestyle='dotted', color='chocolate')
            ax.plot(time_index, temp_cons2, label="Consumer 2", linestyle='dotted', color='slategrey')
            ax.plot(time_index, temp_cons3, label="Consumer 3", linestyle='dotted', color='darkseagreen')
            ax.plot(time_index, avg_cons_temperature, label="Average", color = 'black', linestyle='solid')
            ax.set_xlabel("Date")
            ax.set_ylabel("Temperature [°C]")
            ax.set_title("Consumer Temperature")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/consumer_temperature_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the excess temperature
            fig, ax = plt.subplots()
            ax.plot(time_index, T_cons1_slack.value, label="Consumer 1", color='chocolate')
            ax.plot(time_index, T_cons2_slack.value, label="Consumer 2", color='slategrey')
            ax.plot(time_index, T_cons3_slack.value, label="Consumer 3", color='darkseagreen')
            ax.set_xlabel("Date")
            ax.set_ylabel("Temperature [°C]")
            ax.set_title("Excess Temperature")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/excess_temperature_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Combined plots ---------------------
            # Plot the electricity price and heat value
            fig, ax = plt.subplots()
            ax.plot(time_index, p_el_import, label="Electricity Import", color='slategrey', linestyle='dashed')
            ax.plot(time_index, p_el_export, label="Electricity Export", color='peru', linestyle='dashed')
            ax.plot(time_index, heat_value, label="Heat Value", color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price [CHF/kWh]")
            ax.set_title("Electricity Price and Heat Value")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/electricity_price_heat_value_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the heat demand and electricity price on two separate y axes
            fig, ax1 = plt.subplots()
            ax1.plot(time_index, heat_demand, label="Heat Demand", color='red')
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Heat Demand [kW]")
            ax1.set_title("Heat Demand and Electricity Price")
            ax2 = ax1.twinx()
            ax2.plot(time_index, p_el_import, label="Electricity Price (Import)", color='slategrey', linestyle='dashed')
            ax2.set_ylabel("Price [CHF/kWh]")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/heat_demand_price_year_{name_version}.pdf', format="pdf", bbox_inches="tight")

            ### One week results ###
            for start_date, end_date in weekly_time_frames:
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include the full last day

                # Get the start and end indices using get_indexer
                start_index = time_index.get_indexer([start_ts], method='nearest')[0]
                end_index = time_index.get_indexer([end_ts], method='nearest')[0]

                # Electricity results ---------------------
                # Plot the electricity total demand
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], electricity_for_heat_demand[start_index:end_index], label="Electricity for heat", color='indianred')
                ax.plot(time_index[start_index:end_index], electricity_end_demand[start_index:end_index], label="Electricity end demand", color='goldenrod')
                ax.set_xlabel("Date")
                ax.set_ylabel("Power [kW]")
                ax.set_title("Electricity Demand")
                ax.legend()
                # Make xticks at 45 degree angle
                plt.xticks(rotation=45)
                if save_images:
                    plt.savefig(f'{folder_path}/electricity_demand_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Stacked plot of electricity demand
                fig, ax = plt.subplots()
                ax.stackplot(time_index[start_index:end_index], electricity_for_heat_demand[start_index:end_index], electricity_end_demand[start_index:end_index], labels=["Electricity for heat", "Electricity end demand"], colors=['indianred', 'goldenrod'])
                ax.set_xlabel("Date")
                ax.set_ylabel("Power [kW]")
                ax.set_title("Electricity Demand")
                ax.legend()
                # Make xticks at 45 degree angle
                plt.xticks(rotation=45)
                if save_images:
                    plt.savefig(f'{folder_path}/electricity_demand_stacked_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot the electricity generation
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], electricity_generated[start_index:end_index], label="Electricity Generation", color='limegreen')
                ax.set_xlabel("Date")
                ax.set_ylabel("Power [kW]")
                ax.set_title("Electricity Generation")
                ax.legend()
                # Make xticks at 45 degree angle
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/electricity_generation_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot stacked electricity generation
                fig, ax = plt.subplots()
                ax.stackplot(time_index[start_index:end_index], electricity_pv_produced[start_index:end_index], electricity_fc_produced[start_index:end_index], labels=["PV", "Fuel Cell"], colors=['gold', 'orangered'])
                ax.set_xlabel("Date")
                ax.set_ylabel("Power [kW]")
                ax.set_title("Electricity Generation")
                ax.legend()
                # Make xticks at 45 degree angle
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/electricity_generation_stacked_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot net electricity imports
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], electricity_net_imported[start_index:end_index], label="Net electricity [kW]") 
                ax.set_xlabel("Date")
                ax.set_ylabel("Power [kW]")
                ax.set_title("Net Electricity")
                ax.legend()
                # Make xticks at 45 degree angle
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/net_electricity_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot electricity import, export and electricity prices on separate y axes
                fig, ax1 = plt.subplots()
                ax1.plot(time_index[start_index:end_index], electricity_imported[start_index:end_index], label="Import electricity", color='midnightblue', marker='o')
                ax1.plot(time_index[start_index:end_index], electricity_exported[start_index:end_index], label="Export electricity", color='darkorange', marker='x')
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Power [kW]")
                ax1.set_title("Electricity Import and Export")
                ax1.tick_params(axis='x', rotation=45)
                ax2 = ax1.twinx()
                ax2.plot(time_index[start_index:end_index], p_el_import[start_index:end_index], label="Electricity Price (Import)", color='slategrey', linestyle='dashed')
                ax2.plot(time_index[start_index:end_index], p_el_export[start_index:end_index], label="Electricity Price (Export)", color='peru', linestyle='dashed')
                ax2.set_ylabel("Price [CHF/kWh]")
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                # Make xticks at 45 degree angle
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/electricity_import_export_price_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")
                
                # Plot electricity import and import prices
                fig, ax1 = plt.subplots()
                ax1.plot(time_index[start_index:end_index], electricity_imported[start_index:end_index], label="Import electricity [kW]", color='midnightblue', marker='o')
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Power [kW]")
                ax1.tick_params(axis='x', rotation=45)
                ax2 = ax1.twinx()
                ax2.plot(time_index[start_index:end_index], p_el_import[start_index:end_index], label="Electricity Price (Import)", color='slategrey', linestyle='dashed')
                ax2.set_ylabel("Price [CHF/kWh]")
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                # Make xticks at 45 degree angle
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/electricity_import_price_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Heat results ---------------------
                # Plot the heat demand
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], q_cons1.value[start_index:end_index], label="Consumer 1", color='chocolate', linestyle='dotted')
                ax.plot(time_index[start_index:end_index], q_cons2.value[start_index:end_index], label="Consumer 2", color='slategrey', linestyle='dotted')
                ax.plot(time_index[start_index:end_index], q_cons3.value[start_index:end_index], label="Consumer 3", color='darkseagreen', linestyle='dotted')
                ax.plot(time_index[start_index:end_index], heat_demand[start_index:end_index], label="Total", color='black', linestyle='solid', marker='o')
                ax.set_xlabel("Date")
                ax.set_ylabel("Heat [kW]")
                ax.set_title("Heat Demand")
                ax.legend()
                # Make xticks at 45 degree angle
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/heat_demand_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot the heat supply
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], heat_h2_used[start_index:end_index], label="Hydrogen heat", color='limegreen')
                ax.plot(time_index[start_index:end_index], heat_hp_used[start_index:end_index], label="Heat Pump", color='darkkhaki')
                ax.plot(time_index[start_index:end_index], heat_supply[start_index:end_index], label="Total", color='black', linestyle='solid', marker='o')
                ax.set_xlabel("Date")
                ax.set_ylabel("Heat [kW]")
                ax.set_title("Heat Supply")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/heat_supply_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot the stacked heat supply
                fig, ax = plt.subplots()
                ax.stackplot(time_index[start_index:end_index], heat_el_used[start_index:end_index], heat_fc_used[start_index:end_index], heat_hp_used[start_index:end_index], labels=["Electrolyser", "Fuel Cell", "Heat Pump"], colors=['royalblue', 'orangered', 'darkkhaki'])
                ax.set_xlabel("Date")
                ax.set_ylabel("Heat [kW]")
                ax.set_title("Heat Supply")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/heat_supply_stacked_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")
                
                # Plot the total heat demand and heat value on two separate y axes
                fig, ax1 = plt.subplots()
                ax1.plot(time_index[start_index:end_index], heat_demand[start_index:end_index], label="Heat Demand", color='black', marker='o')
                ax.set_xlabel("Date")
                ax1.set_ylabel("Heat Demand [kW]")
                ax1.set_title("Heat Demand and Value")
                ax1.tick_params(axis='x', rotation=45)
                ax2 = ax1.twinx()
                ax2.plot(time_index[start_index:end_index], heat_value[start_index:end_index], label="Heat Value", color='red', linestyle='dashed', marker='x')
                ax2.set_ylabel("Heat Value [CHF/kWh]")
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/heat_demand_value_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")
                
                # Hydrogen results ---------------------
                #Plot the hydrogen in, out and storage level
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], hydrogen_imported[start_index:end_index] + hydrogen_production[start_index:end_index], label="In", color='slategrey', linestyle='dotted', marker='o')
                ax.plot(time_index[start_index:end_index], hydrogen_exported[start_index:end_index] + hydrogen_consumption[start_index:end_index], label="Out", color='peru', linestyle='dotted', marker='x')
                # Plot storage level colored in
                ax.fill_between(time_index[start_index:end_index], hydrogen_storage_level[start_index:end_index], color='cadetblue', alpha=0.5, label="Storage")
                ax.set_xlabel("Date")
                ax.set_ylabel("Hydrogen [kg]")
                ax.set_title("Hydrogen In, Out and Storage Level")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/hydrogen_in_out_storage_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches = "tight")

                # Temperature results ---------------------
                # Plot the consumer temperature
                avg_cons_temperature = (temp_cons1 + temp_cons2 + temp_cons3) / n_consumers
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], temp_cons1[start_index:end_index], label="Consumer 1", linestyle='dotted', color='chocolate')
                ax.plot(time_index[start_index:end_index], temp_cons2[start_index:end_index], label="Consumer 2", linestyle='dotted', color='slategrey')
                ax.plot(time_index[start_index:end_index], temp_cons3[start_index:end_index], label="Consumer 3", linestyle='dotted', color='darkseagreen')
                ax.plot(time_index[start_index:end_index], avg_cons_temperature[start_index:end_index], label="Average", color = 'black', linestyle='solid', marker='o')
                ax.set_xlabel("Date")
                ax.set_ylabel("Temperature [°C]")
                ax.set_title("Consumer Temperature")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/consumer_temperature_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Combined plots ---------------------
                # Plot the electricity price and heat value
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], p_el_import[start_index:end_index], label="Electricity Import", color='slategrey', linestyle='dashed')
                ax.plot(time_index[start_index:end_index], p_el_export[start_index:end_index], label="Electricity Export", color='peru', linestyle='dashed')
                ax.plot(time_index[start_index:end_index], heat_value[start_index:end_index], label="Heat Value", color='red')
                ax.set_xlabel("Date")
                ax.set_ylabel("Price [CHF/kWh]")
                ax.set_title("Electricity Price and Heat Value")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/electricity_price_heat_value_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot the heat demand and electricity price on two separate y axes
                fig, ax1 = plt.subplots()
                ax1.plot(time_index[start_index:end_index], heat_demand[start_index:end_index], label="Heat Demand", color='red')
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Heat Demand [kW]")
                ax1.set_title("Heat Demand and Electricity Price")
                ax2 = ax1.twinx()
                ax2.plot(time_index[start_index:end_index], p_el_import[start_index:end_index], label="Electricity Price (Import)", color='slategrey', linestyle='dashed')
                ax2.plot(time_index[start_index:end_index], p_el_export[start_index:end_index], label="Electricity Price (Export)", color='peru', linestyle='dashed')
                ax2.set_ylabel("Price [CHF/kWh]")
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/heat_demand_price_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")
                
                # Plot the avg consumer temperature and heat value on two separate y axes
                fig, ax1 = plt.subplots()
                ax1.plot(time_index[start_index:end_index], avg_cons_temperature[start_index:end_index], label="Average Consumer Temperature", color='black')
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Temperature [°C]")
                ax1.set_title("Average Consumer Temperature and Heat Value")
                ax1.tick_params(axis='x', rotation=45)
                ax2 = ax1.twinx()
                ax2.plot(time_index[start_index:end_index], heat_value[start_index:end_index], label="Heat Value", color='red', linestyle='dashed', marker='x')
                ax2.set_ylabel("Heat Value [CHF/kWh]")
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/avg_consumer_temperature_heat_value_week_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")


                
            
            ### One day results ###
            for start_date, end_date in daily_time_frames:
                # Convert the dates to timestamps
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date) 

                # Get the start and end indices using get_indexer
                start_index = time_index.get_indexer([start_ts], method='nearest')[0]
                end_index = time_index.get_indexer([end_ts], method='nearest')[0]

                # Electricity results ---------------------
                # Plot the electricity total demand
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], electricity_for_heat_demand[start_index:end_index], label="Electricity for heat", color='indianred')
                ax.plot(time_index[start_index:end_index], electricity_end_demand[start_index:end_index], label="Electricity end demand", color='goldenrod')
                ax.set_xlabel("Date")
                ax.set_ylabel("Power [kW]")
                ax.set_title("Electricity Demand")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    plt.savefig(f'{folder_path}/electricity_demand_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Stacked plot of electricity demand
                fig, ax = plt.subplots()
                ax.stackplot(time_index[start_index:end_index], electricity_for_heat_demand[start_index:end_index], electricity_end_demand[start_index:end_index], labels=["Electricity for heat", "Electricity end demand"], colors=['indianred', 'goldenrod'])
                ax.set_xlabel("Date")
                ax.set_ylabel("Power [kW]")
                ax.set_title("Electricity Demand")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    plt.savefig(f'{folder_path}/electricity_demand_stacked_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot the electricity generation
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], electricity_generated[start_index:end_index], label="Electricity Generation", color='limegreen')
                ax.set_xlabel("Date")
                ax.set_ylabel("Power [kW]")
                ax.set_title("Electricity Generation")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/electricity_generation_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot stacked electricity generation
                fig, ax = plt.subplots()
                ax.stackplot(time_index[start_index:end_index], electricity_pv_produced[start_index:end_index], electricity_fc_produced[start_index:end_index], labels=["PV", "Fuel Cell"], colors=['gold', 'orangered'])
                ax.set_xlabel("Date")
                ax.set_ylabel("Power [kW]")
                ax.set_title("Electricity Generation")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/electricity_generation_stacked_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot net electricity imports
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], electricity_net_imported[start_index:end_index], label="Net electricity [kW]") 
                ax.set_xlabel("Date")
                ax.set_ylabel("Power [kW]")
                ax.set_title("Net Electricity")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/net_electricity_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot electricity import, export and electricity prices on separate y axes
                fig, ax1 = plt.subplots()
                ax1.plot(time_index[start_index:end_index], electricity_imported[start_index:end_index], label="Import electricity", color='midnightblue', marker='o')
                ax1.plot(time_index[start_index:end_index], electricity_exported[start_index:end_index], label="Export electricity", color='darkorange', marker='x')
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Power [kW]")
                ax1.set_title("Electricity Import and Export")
                ax1.tick_params(axis='x', rotation=45)
                ax2 = ax1.twinx()
                ax2.plot(time_index[start_index:end_index], p_el_import[start_index:end_index], label="Electricity Price (Import)", color='slategrey', linestyle='dashed')
                ax2.plot(time_index[start_index:end_index], p_el_export[start_index:end_index], label="Electricity Price (Export)", color='peru', linestyle='dashed')
                ax2.set_ylabel("Price [CHF/kWh]")
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                # Make xticks at 45 degree angle
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/electricity_import_export_price_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")
                
                # Plot electricity import and import prices
                fig, ax1 = plt.subplots()
                ax1.plot(time_index[start_index:end_index], electricity_imported[start_index:end_index], label="Import electricity [kW]", color='midnightblue', marker='o')
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Power [kW]")
                ax1.tick_params(axis='x', rotation=45)
                ax2 = ax1.twinx()
                ax2.plot(time_index[start_index:end_index], p_el_import[start_index:end_index], label="Electricity Price (Import)", color='slategrey', linestyle='dashed')
                ax2.set_ylabel("Price [CHF/kWh]")
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                # Make xticks at 45 degree angle
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/electricity_import_price_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")
                
                # Heat results ---------------------
                # Plot the heat demand
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], q_cons1.value[start_index:end_index], label="Consumer 1", color='chocolate', linestyle='dotted')
                ax.plot(time_index[start_index:end_index], q_cons2.value[start_index:end_index], label="Consumer 2", color='slategrey', linestyle='dotted')
                ax.plot(time_index[start_index:end_index], q_cons3.value[start_index:end_index], label="Consumer 3", color='darkseagreen', linestyle='dotted')
                ax.plot(time_index[start_index:end_index], heat_demand[start_index:end_index], label="Total", color='black', linestyle='solid', marker='o')
                ax.set_xlabel("Date")
                ax.set_ylabel("Heat [kW]")
                ax.set_title("Heat Demand")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/heat_demand_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot the heat supply
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], heat_h2_used[start_index:end_index], label="Hydrogen heat", color='limegreen')
                ax.plot(time_index[start_index:end_index], heat_hp_used[start_index:end_index], label="Heat Pump", color='darkkhaki')
                ax.plot(time_index[start_index:end_index], heat_supply[start_index:end_index], label="Total", color='black', linestyle='solid', marker='o')
                ax.set_xlabel("Date")
                ax.set_ylabel("Heat [kW]")
                ax.set_title("Heat Supply")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/heat_supply_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot the stacked heat supply
                fig, ax = plt.subplots()
                ax.stackplot(time_index[start_index:end_index], heat_el_used[start_index:end_index], heat_fc_used[start_index:end_index], heat_hp_used[start_index:end_index], labels=["Electrolyser", "Fuel Cell", "Heat Pump"], colors=['royalblue', 'orangered', 'darkkhaki'])
                ax.set_xlabel("Date")
                ax.set_ylabel("Heat [kW]")
                ax.set_title("Heat Supply")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/heat_supply_stacked_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")
                
                # Plot the total heat demand and heat value on two separate y axes
                fig, ax1 = plt.subplots()
                ax1.plot(time_index[start_index:end_index], heat_demand[start_index:end_index], label="Heat Demand", color='black', marker='o')
                ax.set_xlabel("Date")
                ax1.set_ylabel("Heat Demand [kW]")
                ax1.set_title("Heat Demand and Value")
                ax1.tick_params(axis='x', rotation=45)
                ax2 = ax1.twinx()
                ax2.plot(time_index[start_index:end_index], heat_value[start_index:end_index], label="Heat Value", color='red', linestyle='dashed', marker='x')
                ax2.set_ylabel("Heat Value [CHF/kWh]")
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/heat_demand_value_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")
                
                # Hydrogen results ---------------------
                #Plot the hydrogen in, out and storage level
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], hydrogen_imported[start_index:end_index] + hydrogen_production[start_index:end_index], label="In", color='slategrey', linestyle='dotted', marker='o')
                ax.plot(time_index[start_index:end_index], hydrogen_exported[start_index:end_index] + hydrogen_consumption[start_index:end_index], label="Out", color='peru', linestyle='dotted', marker='x')
                # Plot storage level colored in
                ax.fill_between(time_index[start_index:end_index], hydrogen_storage_level[start_index:end_index], color='cadetblue', alpha=0.5, label="Storage")
                ax.set_xlabel("Date")
                ax.set_ylabel("Hydrogen [kg]")
                ax.set_title("Hydrogen In, Out and Storage Level")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/hydrogen_in_out_storage_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches = "tight")

                # Temperature results ---------------------
                # Plot the consumer temperature
                avg_cons_temperature = (temp_cons1 + temp_cons2 + temp_cons3) / n_consumers
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], temp_cons1[start_index:end_index], label="Consumer 1", linestyle='dotted', color='chocolate')
                ax.plot(time_index[start_index:end_index], temp_cons2[start_index:end_index], label="Consumer 2", linestyle='dotted', color='slategrey')
                ax.plot(time_index[start_index:end_index], temp_cons3[start_index:end_index], label="Consumer 3", linestyle='dotted', color='darkseagreen')
                ax.plot(time_index[start_index:end_index], avg_cons_temperature[start_index:end_index], label="Average", color = 'black', linestyle='solid', marker='o')
                ax.set_xlabel("Date")
                ax.set_ylabel("Temperature [°C]")
                ax.set_title("Consumer Temperature")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/consumer_temperature_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Combined plots ---------------------
                # Plot the electricity price and heat value
                fig, ax = plt.subplots()
                ax.plot(time_index[start_index:end_index], p_el_import[start_index:end_index], label="Electricity Import", color='slategrey', linestyle='dashed')
                ax.plot(time_index[start_index:end_index], p_el_export[start_index:end_index], label="Electricity Export", color='peru', linestyle='dashed')
                ax.plot(time_index[start_index:end_index], heat_value[start_index:end_index], label="Heat Value", color='red')
                ax.set_xlabel("Date")
                ax.set_ylabel("Price [CHF/kWh]")
                ax.set_title("Electricity Price and Heat Value")
                ax.legend()
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/electricity_price_heat_value_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

                # Plot the heat demand and electricity price on two separate y axes
                fig, ax1 = plt.subplots()
                ax1.plot(time_index[start_index:end_index], heat_demand[start_index:end_index], label="Heat Demand", color='red')
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Heat Demand [kW]")
                ax1.set_title("Heat Demand and Electricity Price")
                ax1.tick_params(axis='x', rotation=45)
                ax2 = ax1.twinx()
                ax2.plot(time_index[start_index:end_index], p_el_import[start_index:end_index], label="Electricity Price (Import)", color='slategrey', linestyle='dashed')
                ax2.plot(time_index[start_index:end_index], p_el_export[start_index:end_index], label="Electricity Price (Export)", color='peru', linestyle='dashed')
                ax2.set_ylabel("Price [CHF/kWh]")
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                plt.xticks(rotation=45)
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/heat_demand_price_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")
                
                # Plot the avg consumer temperature and heat value on two separate y axes
                fig, ax1 = plt.subplots()
                ax1.plot(time_index[start_index:end_index], avg_cons_temperature[start_index:end_index], label="Average Consumer Temperature", color='black')
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Temperature [°C]")
                ax1.set_title("Average Consumer Temperature and Heat Value")
                ax1.tick_params(axis='x', rotation=45)
                ax2 = ax1.twinx()
                ax2.plot(time_index[start_index:end_index], heat_value[start_index:end_index], label="Heat Value", color='red', linestyle='dashed', marker='x')
                ax2.set_ylabel("Heat Value [CHF/kWh]")
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                if save_images:
                    # save figure
                    plt.savefig(f'{folder_path}/avg_consumer_temperature_heat_value_day_{start_date}_{name_version}.pdf', format="pdf", bbox_inches="tight")

            if plot_results:
                plt.show()

if __name__ == "__main__":
    main()
