# decentralisedv3 makes use of the ADMM algorithm to solve the decentralised optimisation problem
# The problem is divided into 5 subproblems: hydrogen system, heat pump, consumer 1, consumer 2 and consumer 3
# The difference to decentralisedv1 is that the optimisation problems are built once, and parameters are updated
# The difference to decentralisedv2 is that the problem is phrased as Exchange ADMM, where the average of the primal variables is used
# The difference to decentralisedv3 is that the penalty parameter is updated at every iteration
# The difference to decentralisedv3 is that the temperature constraints are relaxed to allow for a feasible solution
# The difference to decentralisedv3 is also that hydrogen is exported

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parametersv1 import *
import os
import additional_functions as af

def main():
        
    #-----
    # Values from centralised to plot
    min_cost_centralised    = 228.05 # Minimum cost for the system from the centralised optimisation

    #------
    # Configuration of this run
    af.configure_plots(style='fancy')
    import_price = 'bkw' # 'groupe_e' or 'bkw' or 'spot'
    tariff_name  = 'green' # groupe_e: 'vario_plus', 'vario_grid', 'dt_plus' | bwk: 'green', 'blue', 'grey' | spot: 'spot' or 'plus_tariff'
    thermal_inertia = 'high' # 'low', 'medium', or 'high' or 'different'
    save_results = True
    plot_results = False
    save_images  = True
    name_version = f"decentralisedv5-{import_price}_{tariff_name}-{thermal_inertia}_inertia" # Convention: scriptname-tariff-thermal_inertia
    constant_slack_cost = 1e0 # At 1e-1 it exceeds the temperature at some points in winter. At 1e0 it does not.NOTE: This is a constant cost for the slack variable, a very high value (>1e3) may affect the convergence of the algorithm
    # Penalty parameter values for the ADMM algorithm
    # rho_values     = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]     # Multiple values of the penalty parameter
    rho_values     = [0.001, 0.002,0.005,0.01]     # Initial value of the penalty parameter
    #------

    # Create results folders if they do not exist
    folder_version = f"results/{name_version}"
    if not os.path.exists(f"{folder_version}"):
        os.makedirs(f"{folder_version}")
    else:
        print(f"{folder_version} folder already exists.")

    # Define the ADMM algorithm parameters
    max_iter        = 50   # Maximum number of iterations
    tolerance_primal= 1e-4 # Tolerance of the primal residual
    tolerance_dual  = 1e-4 # Tolerance of the dual residual
    n_agents        = 6         # Number of agents - h2, hp, cons1, cons2, cons3, slack

    # Import data
    p_el_export = af.get_spot_prices()                          # Electricity export price [CHF/kWh]
    if import_price == 'groupe_e':
        p_el_import = af.get_groupe_e_tariff_data(resolution='hourly', tariff=tariff_name)       # Electricity import price [CHF/kWh]
    elif import_price == 'bkw':
        p_el_import = af.get_bkw_tariff_data(tariff=tariff_name)       # Electricity import price [CHF/kWh]
    elif import_price == 'spot':
        if tariff_name == 'spot':
            p_el_import = af.get_spot_prices()                          # Electricity import price [CHF/kWh]
        elif tariff_name == 'plus_tariff':
            p_el_import = af.get_spot_prices() + 0.13                   # Electricity import price [CHF/kWh] 13 cents per kWh -> mentioned by Binod
    else:
        raise ValueError("Error: Invalid type of electricity price")
    T_amb       = af.get_temperature_data()                     # Ambient temperature [°C]
    price_h2    = af.get_hydrogen_price() * np.ones(T)          # Hydrogen price [CHF/kgH2] - 3.3: https://data.sccer-jasm.ch/import-prices/2020-08-01/
    q_gain      = np.zeros(T)                                   # Heat gain from sun irradiation [kW] - TODO: add real data based on solar irradiation
    
    # Select the thermal inertia
    if thermal_inertia == 'low':
        R_cons1 = R_cons2 = R_cons3 = R_cons_low
        C_cons1 = C_cons2 = C_cons3 = C_cons_low
    elif thermal_inertia == 'medium':
        R_cons1 = R_cons2 = R_cons3 = R_cons_medium
        C_cons1 = C_cons2 = C_cons3 = C_cons_medium
    elif thermal_inertia == 'high':
        R_cons1 = R_cons2 = R_cons3 = R_cons_high
        C_cons1 = C_cons2 = C_cons3 = C_cons_high
    elif thermal_inertia == 'different':
        R_cons1 = R_cons_low
        C_cons1 = C_cons_low
        R_cons2 = R_cons_medium
        C_cons2 = C_cons_medium
        R_cons3 = R_cons_high
        C_cons3 = C_cons_high
    else:
        raise ValueError("Error: Invalid type of thermal inertia")

    # Simplifying parameters
    alpha_cons1= np.exp(-dt/(R_cons1*C_cons1))    # Decay term of consumer 1 [-]
    alpha_cons2= np.exp(-dt/(R_cons2*C_cons2))    # Decay term of consumer 2 [-]
    alpha_cons3= np.exp(-dt/(R_cons3*C_cons3))    # Decay term of consumer 3 [-]


    for initial_rho in rho_values:
        rho     = cp.Parameter(nonneg=True) # Penalty parameter
        # Define the decision variables
        l_h2    = cp.Variable(T,nonneg=True) # load of hydrogen system [kW]
        l_hp    = cp.Variable(T,nonneg=True) # load of heat pump [kW]
        p_exp   = cp.Variable(T,nonneg=True) # Export electricity [kW]
        h2_imp  = cp.Variable(T,nonneg=True) # Import hydrogen [kg]
        h2_exp  = cp.Variable(T,nonneg=True) # Export hydrogen [kg]
        q_slack = cp.Variable(T, nonneg=True)# Slack variable for heat balance

        # Define the dependant variables
        # Power
        l_el    = cp.Variable(T,nonneg=True) # load of electrolyser [kW]
        l_co    = cp.Variable(T,nonneg=True) # load of compressor [kW]
        p_fc    = cp.Variable(T,nonneg=True) # power generated from fuel cell [kW]
        # Heat
        q_h2    = cp.Variable(T,nonneg=True) # heat used from hydrogen system [kW]
        q_el    = cp.Variable(T,nonneg=True) # heat used from electrolyser [kW]
        q_gen_el= cp.Variable(T,nonneg=True) # heat generated from electrolyser [kW]
        q_was_el= cp.Variable(T,nonneg=True) # heat wasted from electrolyser [kW]
        q_fc    = cp.Variable(T,nonneg=True) # heat used from fuel cell [kW]
        q_gen_fc= cp.Variable(T,nonneg=True) # heat generated from fuel cell [kW]
        q_was_fc= cp.Variable(T,nonneg=True) # heat wasted from fuel cell [kW]
        q_hp    = cp.Variable(T,nonneg=True) # heat used from heat pump [kW]
        q_cons1 = cp.Variable(T,nonneg=True) # heat demand from consumer 1 [kW]
        q_cons2 = cp.Variable(T,nonneg=True) # heat demand from consumer 2 [kW]
        q_cons3 = cp.Variable(T,nonneg=True) # heat demand from consumer 3 [kW]
        # Hydrogen
        h2_prod = cp.Variable(T,nonneg=True) # hydrogen produced [kg]
        h2_fc   = cp.Variable(T,nonneg=True) # hydrogen used in fuel cell [kg]
        h2_sto  = cp.Variable(T,nonneg=True) # hydrogen stored [kg]
        # Temperature
        T_cons1 = cp.Variable(T) # temperature of consumer 1 [°C]
        T_cons2 = cp.Variable(T) # temperature of consumer 2 [°C]
        T_cons3 = cp.Variable(T) # temperature of consumer 3 [°C]
        # Define temperature slack variables
        T_cons1_slack = cp.Variable(T, nonneg=True)      # Slack variable for consumer 1 temperature
        T_cons2_slack = cp.Variable(T, nonneg=True)      # Slack variable for consumer 2 temperature
        T_cons3_slack = cp.Variable(T, nonneg=True)      # Slack variable for consumer 3 temperature
        cost_slack    = constant_slack_cost * np.ones(T) # Cost of slack variable

        # Define parameters to be exchanged and updated through iterations
        lambda_fixed            = cp.Parameter(T)
        q_h2_fixed              = cp.Parameter(T)
        q_hp_fixed              = cp.Parameter(T)
        q_cons1_fixed           = cp.Parameter(T)
        q_cons2_fixed           = cp.Parameter(T)
        q_cons3_fixed           = cp.Parameter(T)
        q_slack_fixed           = cp.Parameter(T)
        q_average               = cp.Parameter(T)

        # Define functions to build optimisation problems`
        # Hydrogen system
        def hydrogen_system_optimisation():
            """
            hydrogen_system_optimisation defines the optimisation problem for the hydrogen system
            """
            # Define the cost function
            cost_h2             = p_el_import.T @ l_h2 - p_el_export.T @ p_exp + price_h2.T @ h2_imp - price_h2.T @ h2_exp

            # Define the constraints
            hydrogen_network_balance = [l_h2        == l_el + l_co,
                                        q_h2        == q_el + q_fc,
                                        q_gen_el    == q_el + q_was_el,
                                        q_gen_fc    == q_fc + q_was_fc,
                                        p_exp       == p_fc]
            # Define electrolyser constraints
            electrolyser = [l_el        <= l_el_max,
                            l_el        >= 0,
                            h2_prod     == eff_el_h2 * l_el / HHV_H2,
                            q_gen_el    == (1 - eff_el_h2) * eff_el_th * l_el]
            # Define fuel cell constraints
            fuel_cell = [p_fc           <= p_fc_max,
                        p_fc           >= 0,
                        h2_fc          == p_fc / (eff_fc_h2 * HHV_H2),
                        q_gen_fc       == (1 - eff_fc_h2) * eff_fc_th * p_fc]
            # Define compressor constraints
            compressor = [l_co          <= l_co_max,
                        l_co          >= 0,
                        l_co          == k_co * h2_prod]
            # Define storage constraints
            storage = [h2_sto[0]        == h2_sto_max/2,
                    h2_sto[T-1]      == h2_sto_max/2,
                    h2_sto           <= h2_sto_max,
                    h2_sto           >= 0,
                    h2_imp           <= h2_imp_max,
                    h2_exp           <= h2_exp_max,
                    h2_sto[1:T]      == h2_sto[0:T-1] + h2_sto_eff * h2_prod[0:T-1] - (1/h2_sto_eff) * h2_fc[0:T-1] + h2_sto_eff * h2_imp[0:T-1] - (1/h2_sto_eff) * h2_exp[0:T-1]]
            
            # Define augmented lagrangian
            augm_lagr_h2        = cost_h2 - lambda_fixed.T @ q_h2 + rho/2 * cp.sum_squares(n_agents * q_average + q_h2_fixed - q_h2)
            constraints_h2      = hydrogen_network_balance + electrolyser + fuel_cell + compressor + storage
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
            cost_hp             = p_el_import.T @ l_hp
            # Define heat pump constraints
            heat_pump = [q_hp   <= q_hp_max,
                        q_hp    >= 0,
                        q_hp    == COP_hp * l_hp]
            # Define augmented lagrangian
            augm_lagr_hp        = cost_hp - lambda_fixed.T @ q_hp + rho/2 * cp.sum_squares(n_agents * q_average + q_hp_fixed - q_hp)
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
            # Define heat consumer heat constraints
            consumer_1 = [T_cons1       <= T_cons1_max + T_cons1_slack,
                        T_cons1       >= T_cons1_min,
                        T_cons1[0]    == (T_cons1_max + T_cons1_min)/2,
                        T_cons1[1:T]  == alpha_cons1 * T_cons1[0:T-1] + (1-alpha_cons1) * (T_amb[0:T-1] + R_cons1 * (q_cons1[0:T-1] + q_gain[0:T-1]))]
            # Define augmented lagrangian
            augm_lagr_cons1         = cost_slack.T @ T_cons1_slack + lambda_fixed.T @ q_cons1 + rho/2 * cp.sum_squares(n_agents * q_average - q_cons1_fixed + q_cons1)
            constraints_cons1       = consumer_1
            objective_cons1         = cp.Minimize(augm_lagr_cons1)
            # Create the problem
            problem_cons1 = cp.Problem(objective_cons1, constraints_cons1)
            return problem_cons1

        # Heat consumer 2
        def consumer_2_optimisation():
            """
            consumer_2_optimisation defines the optimisation problem for consumer 2
            """
            # Define heat consumer heat constraints
            consumer_2 = [T_cons2       <= T_cons2_max + T_cons2_slack,
                        T_cons2       >= T_cons2_min,
                        T_cons2[0]    == (T_cons2_max + T_cons2_min)/2,
                        T_cons2[1:T]  == alpha_cons2 * T_cons2[0:T-1] + (1-alpha_cons2) * (T_amb[0:T-1] + R_cons2 * (q_cons2[0:T-1] + q_gain[0:T-1]))]
            # Define augmented lagrangian
            augm_lagr_cons2     = cost_slack.T @ T_cons2_slack + lambda_fixed.T @ q_cons2 + rho/2 * cp.sum_squares(n_agents * q_average - q_cons2_fixed + q_cons2)
            constraints_cons2   = consumer_2
            objective_cons2     = cp.Minimize(augm_lagr_cons2)
            # Create the problem
            problem_cons2 = cp.Problem(objective_cons2, constraints_cons2)
            return problem_cons2

        # Heat consumer 3
        def consumer_3_optimisation():
            """
            consumer_3_optimisation defines the optimisation problem for consumer 3
            """
            # Define heat consumer heat constraints
            consumer_3 = [T_cons3       <= T_cons3_max + T_cons3_slack,
                        T_cons3       >= T_cons3_min,
                        T_cons3[0]    == (T_cons3_max + T_cons3_min)/2,
                        T_cons3[1:T]  == alpha_cons3 * T_cons3[0:T-1] + (1-alpha_cons3) * (T_amb[0:T-1] + R_cons3 * (q_cons3[0:T-1] + q_gain[0:T-1]))]
            # Define augmented lagrangian
            augm_lagr_cons3     = cost_slack.T @ T_cons3_slack + lambda_fixed.T @ q_cons3 + rho/2 * cp.sum_squares(n_agents * q_average - q_cons3_fixed + q_cons3)
            constraints_cons3   = consumer_3
            objective_cons3     = cp.Minimize(augm_lagr_cons3)
            # Create the problem
            problem_cons3 = cp.Problem(objective_cons3, constraints_cons3)
            return problem_cons3
        
        # Slack heat balance
        def slack_heat_optimisation():
            """
            slack_heat_optimisation defines the optimisation problem for the slack heat balance
            """
            # Define augmented lagrangian
            augm_lagr_slack     = lambda_fixed.T @ q_slack + rho/2 * cp.sum_squares(n_agents * q_average - q_slack_fixed + q_slack)
            constraints_slack   = [0 == 0]
            objective_slack     = cp.Minimize(augm_lagr_slack)
            # Create the problem
            problem_slack = cp.Problem(objective_slack, constraints_slack)
            return problem_slack

        # Define functions to return the optimal values
        # Hydrogen system
        def hydrogen_system_optimal_value(problem_h2):
            """
            hydrogen_system_optimal_value return the optimal heat produced by the hydrogen system
            """
            # Solve the problem
            problem_h2.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_h2.value

        # Heat pump
        def heat_pump_optimal_value(problem_hp):
            """
            heat_pump_optimal_value obtains the optimal heat produced by the hydrogen system
            """
            # Solve the problem
            problem_hp.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_hp.value

        # Heat consumer 1
        def consumer_1_optimal_value(problem_cons1):
            """
            consumer_1_optimal_value obtains the optimal heat consumed by consumer 1
            """
            # Solve the problem
            problem_cons1.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_cons1.value

        # Heat consumer 2
        def consumer_2_optimal_value(problem_cons2):
            """
            consumer_2_optimal_value obtains the optimal heat consumed by consumer 2
            """
            # Solve the problem
            problem_cons2.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_cons2.value

        # Heat consumer 3
        def consumer_3_optimal_value(problem_cons3):
            """
            consumer_3_optimal_value obtains the optimal heat consumed by consumer 3
            """
            # Solve the problem
            problem_cons3.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_cons3.value
        
        # Slack heat balance
        def slack_heat_optimal_value(problem_slack):
            """
            slack_heat_optimal_value obtains the optimal slack heat balance
            """
            # Solve the problem
            problem_slack.solve(solver=cp.GUROBI, verbose=False, ignore_dpp=True)
            return q_slack.value
        
        # Calculating average heat to be exchanged
        def average_heat_value():
            """
            average_heat_value calculates the average heat value to be exchanged
            """
            return (q_cons1_fixed.value + q_cons2_fixed.value + q_cons3_fixed.value + q_slack_fixed.value - q_h2_fixed.value - q_hp_fixed.value) / n_agents

        # Initial values
        lambda_fixed.value            = np.zeros(T) # Lagrangian variable [CHF/kWh]
        q_h2_fixed.value              = np.zeros(T) # Initial value of the hydrogen system heat
        q_hp_fixed.value              = np.zeros(T) # Initial value of the heat pump heat
        q_cons1_fixed.value           = np.zeros(T) # Initial value of the consumer 1 heat
        q_cons2_fixed.value           = np.zeros(T) # Initial value of the consumer 2 heat
        q_cons3_fixed.value           = np.zeros(T) # Initial value of the consumer 3 heat
        q_slack_fixed.value           = np.zeros(T) # Initial value of the slack heat balance
        rho.value                     = initial_rho # Initial value of the penalty parameter
        q_average.value               = np.zeros(T) # Average heat value

        # Build the optimisation problems
        problem_h2      = hydrogen_system_optimisation()
        problem_hp      = heat_pump_optimisation()
        problem_cons1   = consumer_1_optimisation()
        problem_cons2   = consumer_2_optimisation()
        problem_cons3   = consumer_3_optimisation()
        problem_slack   = slack_heat_optimisation()

        # Storing values over iterations
        lambda_storage      = np.zeros((T,max_iter))
        primal_gap_storage  = np.zeros((T,max_iter))
        primal_norm_storage = np.zeros(max_iter)
        cost_storage        = np.zeros(max_iter)

        for iter in range(max_iter):
            q_average.value = average_heat_value()

            # Update the hydrogen system
            q_h2.value          = hydrogen_system_optimal_value(problem_h2)
            q_h2_fixed.value    = q_h2.value

            q_average.value = average_heat_value()

            # Update the heat pump
            q_hp.value          = heat_pump_optimal_value(problem_hp)
            q_hp_fixed.value    = q_hp.value

            q_average.value = average_heat_value()

            # Update the heat consumer 1
            q_cons1.value       = consumer_1_optimal_value(problem_cons1)
            q_cons1_fixed.value = q_cons1.value

            q_average.value = average_heat_value()

            # Update the heat consumer 2
            q_cons2.value       = consumer_2_optimal_value(problem_cons2)
            q_cons2_fixed.value = q_cons2.value

            q_average.value = average_heat_value()

            # Update the heat consumer 3
            q_cons3.value       = consumer_3_optimal_value(problem_cons3)
            q_cons3_fixed.value = q_cons3.value

            q_average.value = average_heat_value()

            # Update the slack heat balance
            q_slack.value       = slack_heat_optimal_value(problem_slack)
            q_slack_fixed.value = q_slack.value

            q_average.value = average_heat_value()

            # Residual calculation
            primal_residual = q_cons1.value + q_cons2.value + q_cons3.value + q_slack.value - (q_h2.value + q_hp.value)     # Difference between heat consumption and production at every timestep [kWh]
            total_cost      = p_el_import.T @ (l_h2.value + l_hp.value) - p_el_export.T @ p_exp.value + price_h2.T @ h2_imp.value - price_h2.T @ h2_exp.value # Total cost [CHF]

            # Storing values
            lambda_storage[:,iter]          = lambda_fixed.value
            primal_gap_storage[:,iter]      = primal_residual
            primal_norm_storage[iter]       = np.linalg.norm(primal_residual, ord = 1)
            cost_storage[iter]              = total_cost

            # Update the dual variable - heat value
            lambda_fixed.value      = lambda_fixed.value + rho.value * (primal_residual)
            dual_residual           = lambda_fixed.value - lambda_storage[:,iter]

            # Norms of the residuals
            norm_primal_residual = np.linalg.norm(primal_residual, ord = 1)
            norm_dual_residual   = np.linalg.norm(dual_residual, ord = 1)

            # Update penalty parameter
            # rho.value = 2 * rho.value

            # Check for convergence
            # if all(abs(primal_residual)) < tolerance_primal and all(abs(dual_residual)) < tolerance_dual:
            #     break

            print("Iteration: ", iter," finished.\n")
            print("Current norm of primal residual: ", norm_primal_residual," \n")
            print("Absolute change in dual this iteration: ", norm_dual_residual," \n")
            print("Current total cost value: ", total_cost)


        # Processing of results
        # Main results
        total_cost          = total_cost                    # scalar 
        heat_value          = lambda_fixed.value            # vector
        average_heat_value  = np.average(heat_value)        # scalar
        total_expenses      = p_el_import.T @ l_h2.value + p_el_import.T @ l_hp.value + price_h2.T @ h2_imp.value # scalar
        total_revenue       = p_el_export.T @ p_exp.value + price_h2.T @ h2_exp.value # scalar
        # Cost results
        electricity_expenses    = p_el_import.T @ l_h2.value + p_el_import.T @ l_hp.value # scalar
        electricity_revenue     = p_el_export.T @ p_exp.value # scalar
        hydrogen_expenses       = price_h2.T @ h2_imp.value   # scalar
        hydrogen_revenue        = price_h2.T @ h2_exp.value   # scalar
        expenses_h2_system      = p_el_import.T @ l_h2.value + hydrogen_expenses  # scalar
        revenue_h2_system       = p_el_export.T @ p_exp.value + hydrogen_revenue  # scalar
        expenses_hp_system      = p_el_import.T @ l_hp.value   # scalar
        heat_cost_cons1         = heat_value.T @ q_cons1.value # scalar
        heat_cost_cons2         = heat_value.T @ q_cons2.value # scalar
        heat_cost_cons3         = heat_value.T @ q_cons3.value # scalar
        # Heat results
        heat_consumption    = q_cons1.value + q_cons2.value + q_cons3.value # vector
        heat_supply         = q_h2.value + q_hp.value  # vector
        heat_waste          = q_was_el.value + q_was_fc.value # vector
        heat_h2_used        = q_h2.value # vector
        heat_el_used        = q_el.value # vector
        heat_fc_used        = q_fc.value # vector
        heat_hp_used        = q_hp.value # vector
        total_heat_demand   = np.sum(heat_consumption) # scalar
        total_heat_supply   = np.sum(heat_supply) # scalar
        total_heat_waste    = np.sum(heat_waste) # scalar
        total_heat_h2_used  = np.sum(heat_h2_used) # scalar
        total_heat_el_used  = np.sum(heat_el_used) # scalar
        total_heat_fc_used  = np.sum(heat_fc_used) # scalar
        total_heat_hp_used  = np.sum(heat_hp_used) # scalar
        # Electricity results
        electricity_demand          = l_h2.value + l_hp.value   # vector
        electricity_h2_demand       = l_h2.value                # vector
        electricity_el_demand       = l_el.value                # vector
        electricity_co_demand       = l_co.value                # vector
        electricity_hp_demand       = l_hp.value                # vector
        electricity_supply          = p_fc.value                # vector
        total_electricity_demand    = np.sum(electricity_demand) # scalar
        total_electricity_supply    = np.sum(electricity_supply) # scalar
        total_electricity_h2_demand = np.sum(electricity_h2_demand) # scalar
        total_electricity_el_demand = np.sum(electricity_el_demand) # scalar
        total_electricity_co_demand = np.sum(electricity_co_demand) # scalar
        total_electricity_hp_demand = np.sum(electricity_hp_demand) # scalar
        net_electricity             = total_electricity_demand - total_electricity_supply # scalar
        # Hydrogen results
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
        
        # Decentralised results
        absolute_deviation      = norm_primal_residual # scalar

        # Print main results
        print("Minimum cost: ", total_cost)
        print("Total expenses: ", total_expenses)
        print("Total revenue: ", total_revenue)
        print("Average heat cost: ", average_heat_value)
        print("Total heat demand:" , total_heat_demand)
        print("Total electricity consumed: ", total_electricity_demand)
        print("Total electricity produced: ", total_electricity_supply)

        if save_results:
            # Make sure the results folder exists
            rhoname = str(rho.value).replace(".","_")
            folder_path = f"results/{name_version}/rho_{rhoname}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            else:
                print(folder_path, " folder already exists.")

            # Results over iterations
            results_iterations = pd.DataFrame({"Iteration": range(iter),
                                            "PrimalGap": primal_norm_storage[0:iter],
                                            "Cost": cost_storage[0:iter],
                                            "HeatValue": np.sum(lambda_storage, axis=0)[0:iter]})
            # Save to csv
            results_iterations.to_csv(f"{folder_path}/convergence.csv", index=False)
            
            # Saving the scalar results in a csv file - columns: name, value, unit
            scalar_results = {
                'total_cost': (total_cost, 'CHF'),
                'average_heat_value': (average_heat_value, 'CHF/kWh'),
                'total_expenses': (total_expenses, 'CHF'),
                'total_revenue': (total_revenue, 'CHF'),
                'electricity_expenses': (electricity_expenses, 'CHF'),
                'electricity_revenue': (electricity_revenue, 'CHF'),
                'hydrogen_expenses': (hydrogen_expenses, 'CHF'),
                'hydrogen_revenue': (hydrogen_revenue, 'CHF'),
                'expenses_h2_system': (expenses_h2_system, 'CHF'),
                'revenue_h2_system': (revenue_h2_system, 'CHF'),
                'expenses_hp_system': (expenses_hp_system, 'CHF'),
                'heat_cost_cons1': (heat_cost_cons1, 'CHF'),
                'heat_cost_cons2': (heat_cost_cons2, 'CHF'),
                'heat_cost_cons3': (heat_cost_cons3, 'CHF'),
                'total_heat_demand': (total_heat_demand, 'kWh'),
                'total_heat_supply': (total_heat_supply, 'kWh'),
                'total_heat_waste': (total_heat_waste, 'kWh'),
                'total_heat_h2_used': (total_heat_h2_used, 'kWh'),
                'total_heat_el_used': (total_heat_el_used, 'kWh'),
                'total_heat_fc_used': (total_heat_fc_used, 'kWh'),
                'total_heat_hp_used': (total_heat_hp_used, 'kWh'),
                'total_electricity_demand': (total_electricity_demand, 'kWh'),
                'total_electricity_supply': (total_electricity_supply, 'kWh'),
                'total_electricity_h2_demand': (total_electricity_h2_demand, 'kWh'),
                'total_electricity_el_demand': (total_electricity_el_demand, 'kWh'),
                'total_electricity_co_demand': (total_electricity_co_demand, 'kWh'),
                'total_electricity_hp_demand': (total_electricity_hp_demand, 'kWh'),
                'net_electricity': (net_electricity, 'kWh'),
                'total_hydrogen_used': (total_hydrogen_used, 'kg'),
                'total_hydrogen_produced': (total_hydrogen_produced, 'kg'),
                'total_hydrogen_imported': (total_hydrogen_imported, 'kg'),
                'total_hydrogen_exported': (total_hydrogen_exported, 'kg'),
                'avg_hydrogen_storage': (avg_hydrogen_storage, 'kg'),
                'absolute_deviation': (absolute_deviation, 'kWh')
            }
            # Convert to DataFrame
            scalar_results_df = pd.DataFrame([(key, value[0], value[1]) for key, value in scalar_results.items()], columns=['Metric', 'Value', 'Unit'])
            # Save to CSV
            scalar_results_csv = f"{folder_path}/scalar_results.csv"
            scalar_results_df.to_csv(scalar_results_csv, index=False)

            # Main results
            time_index = pd.date_range(start='2023-01-01 00:00:00', end='2023-12-31 23:00:00', freq='H')
            results = pd.DataFrame({"Time": np.arange(0,T,1),
                                    "PriceImportElectricity": p_el_import,
                                    "PriceExportElectricity": p_el_export,
                                    "PriceHydrogen": price_h2,
                                    "TemperatureAmbient": T_amb,
                                    "HeatValue": heat_value,
                                    "HeatH2": heat_h2_used,
                                    "HeatEl": heat_el_used,
                                    "HeatFC": heat_fc_used,
                                    "HeatHP": heat_hp_used,
                                    "HeatWasteEl": q_was_el.value,
                                    "HeatWasteFC": q_was_fc.value,
                                    "HeatDemand1": q_cons1.value,
                                    "HeatDemand2": q_cons2.value,
                                    "HeatDemand3": q_cons3.value,
                                    "HeatConsumption": heat_consumption,
                                    "HeatAvgConsumption": heat_consumption/3,
                                    "HeatSupply": heat_supply,
                                    "HeatWaste": heat_waste,
                                    "ElectricityDemand": electricity_demand,
                                    "LoadH2System": l_h2.value,
                                    "LoadElectrolyser": l_el.value,
                                    "LoadCompressor": l_co.value,
                                    "LoadHeatPump": l_hp.value,
                                    "ElectricitySupply": electricity_supply,
                                    "ElectricityFCGeneration": p_fc.value,
                                    "NetElectricity": electricity_demand - electricity_supply,
                                    "H2Produced": h2_prod.value,
                                    "H2FCUsed": h2_fc.value,
                                    "H2Stored": h2_sto.value,
                                    "H2Imported": h2_imp.value,
                                    "H2Exported": h2_exp.value,
                                    "TCons1": T_cons1.value,
                                    "TCons2": T_cons2.value,
                                    "TCons3": T_cons3.value,
                                    "TConsAvg": (T_cons1.value + T_cons2.value + T_cons3.value) / 3,
                                    "TCons1Slack": T_cons1_slack.value,
                                    "TCons2Slack": T_cons2_slack.value,
                                    "TCons3Slack": T_cons3_slack.value,
                                    "HeatSlack": q_slack.value,
                                    })
            # Add time_index as a column
            results["DateTime"] = time_index
            # Save to csv
            results.to_csv(f"{folder_path}/time_data_{name_version}.csv", index=False)

            # Plot the results
            ## ----- Plot the convergence results (decentralised version only) -----
            fig, ax = plt.subplots()
            ax.plot(cost_storage[0:iter], label="Cost [CHF]")
            ax.axhline(y = min_cost_centralised, color = 'r', linestyle = '.', label = "Centralised cost")
            ax.set(xlabel="Iteration", 
                ylabel="Cost [CHF]", 
                title="Convergence to Minimum Cost")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/convergence_cost_{name_version}.pdf', format="pdf", bbox_inches="tight")

            fig, ax = plt.subplots()
            ax.plot(primal_norm_storage[0:iter], label="Primal Gap [kWh]")
            ax.set(xlabel="Iteration",
                ylabel="Norm of residual [kWh]",
                title="Convergence to feasible solution")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/convergence_residual_{name_version}.pdf', format="pdf", bbox_inches="tight")


            ## ----- Plot the results over time ----- (same as in centralised version)

       
            # Make hourly time index for the plots
            #### Yearly results ####
            # Electricity results ---------------------
            fig, ax = plt.subplots()
            ax.plot(time_index, electricity_demand, label="Import electricity [kW]")
            ax.set(xlabel="Date", 
                ylabel="Power [kW]", 
                title="Power Import")
            if save_images:
                plt.savefig(f'{folder_path}/power_import_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the power export
            fig, ax = plt.subplots()
            ax.plot(time_index, electricity_supply, label="Export electricity [kW]")
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Power Export")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/power_export_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot net electricity
            fig, ax = plt.subplots()
            ax.plot(time_index, electricity_demand - electricity_supply, label="Net electricity [kW]") 
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Net Electricity")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/net_electricity_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plotting import and export as area plots, net electricity as line plot. Export is negative.
            fig, ax = plt.subplots()
            ax.fill_between(time_index, electricity_demand, 0, label="Import electricity", color='blue', alpha=0.5)
            ax.fill_between(time_index, - electricity_supply, 0, label="Export electricity", color='red', alpha=0.5)
            ax.plot(time_index, electricity_demand - electricity_supply, label="Net electricity", color='black', alpha = 1, linestyle = 'dashed')
            ax.set_xlabel("Date")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Electricity Import and Export")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/electricity_import_export_net_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the electricity price
            fig, ax = plt.subplots()
            ax.plot(time_index, p_el_import, label="Import")
            ax.plot(time_index, p_el_export, label="Export")
            ax.set_xlabel("Date")
            ax.set_ylabel("Electricity Price [CHF/kWh]")
            ax.set_title("Electricity Price")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/electricity_price_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Temperature results ---------------------
            # Plot the ambient temperature
            fig, ax = plt.subplots()
            ax.plot(time_index, T_amb, label="Ambient")
            ax.set_xlabel("Date")
            ax.set_ylabel("Temperature [°C]")
            ax.set_title("Ambient Temperature")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/ambient_temperature_{name_version}.pdf', format="pdf", bbox_inches="tight")

            avg_cons_temperature = (T_cons1.value + T_cons2.value + T_cons3.value) / 3
            # Plot the consumer temperature
            fig, ax = plt.subplots()
            ax.plot(time_index, T_cons1.value, label="Consumer 1", linestyle='dotted')
            ax.plot(time_index, T_cons2.value, label="Consumer 2", linestyle='dotted')
            ax.plot(time_index, T_cons3.value, label="Consumer 3", linestyle='dotted')
            ax.plot(time_index, avg_cons_temperature, label="Average", color = 'black', alpha = 0.7, linestyle='solid')
            ax.set_xlabel("Date")
            ax.set_ylabel("Temperature [°C]")
            ax.set_title("Consumer Temperature")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/consumer_temperature_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the excess temperature
            fig, ax = plt.subplots()
            ax.plot(time_index, np.maximum(0,T_cons1.value - T_cons1_max), label="Consumer 1")
            ax.plot(time_index, np.maximum(0,T_cons2.value - T_cons2_max), label="Consumer 2")
            ax.plot(time_index, np.maximum(0,T_cons3.value - T_cons3_max), label="Consumer 3")
            ax.plot(time_index, np.maximum(0,avg_cons_temperature - T_cons1_max), label="Average", linestyle='solid')
            ax.set_xlabel("Data")
            ax.set_ylabel("Temperature [°C]")
            ax.set_title("Excess Temperature")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/excess_temperature_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Heat results ---------------------
            # Plot the heat demand
            fig, ax = plt.subplots()
            ax.plot(time_index, q_cons1.value, label="Consumer 1 Heat Demand [kW]")
            ax.plot(time_index, q_cons2.value, label="Consumer 2 Heat Demand [kW]")
            ax.plot(time_index, q_cons3.value, label="Consumer 3 Heat Demand [kW]")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Heat Demand [kW]")
            ax.set_title("Heat Demand")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/heat_demand_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the heat value
            fig, ax = plt.subplots()
            ax.plot(time_index, heat_value, label="Heat value [CHF/kWh]")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Heat Value [CHF/kWh]")
            ax.set_title("Heat Value")
            ax.legend()
            if save_images:
                # save figure
                plt.savefig(f'{folder_path}/heat_value_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plotting heat dumped slack
            fig, ax = plt.subplots()
            ax.plot(time_index, q_slack.value, label="Heat Dumped Slack [kW]")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Heat [kW]")
            ax.set_title("Heat Dumped Slack")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/heat_dumped_slack_{name_version}.pdf', format="pdf", bbox_inches="tight")


            # Hydrogen results ---------------------
            # Plot the hydrogen storage
            fig, ax = plt.subplots()
            ax.plot(time_index, hydrogen_storage_level, label="Hydrogen Storage [kg]")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Storage [kg]")
            ax.set_title("Hydrogen Storage")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/hydrogen_storage_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the hydrogen production, consumption, import and export
            fig, ax = plt.subplots()
            ax.plot(time_index, hydrogen_production, label="Hydrogen Production [kg]")
            ax.plot(time_index, hydrogen_consumption, label="Hydrogen Consumption [kg]")
            ax.plot(time_index, hydrogen_imported, label="Hydrogen Import [kg]")
            ax.plot(time_index, hydrogen_exported, label="Hydrogen Export [kg]")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Hydrogen [kg]")
            ax.set_title("Hydrogen Production, Consumption, Import and Export")
            ax.legend()
            if save_images:
                plt.savefig(f'{folder_path}/hydrogen_results_{name_version}.pdf', format="pdf", bbox_inches="tight")

            ### One week results ###
            # One week results
            # Electricity results ---------------------
            fig, ax = plt.subplots()
            ax.plot(time_index, electricity_demand, label="Import electricity [kW]")
            ax.plot(time_index, electricity_supply, label="Export electricity [kW]")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Power [kW]")
            ax.set_title("Power Import and Export")
            ax.legend()
            plt.xticks(rotation=45)
            ax.set_xlim([time_index[0], time_index[167]])
            if save_images:
                plt.savefig(f'{folder_path}/power_import_export_week1_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the electricity price
            fig, ax = plt.subplots()
            ax.plot(time_index, p_el_import, label="Electricity Import Price [CHF/kWh]")
            ax.plot(time_index, p_el_export, label="Electricity Export Price [CHF/kWh]")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Price [CHF/kWh]")
            ax.set_title("Electricity Price")
            ax.legend()
            plt.xticks(rotation=45)
            ax.set_xlim([time_index[0], time_index[167]])
            if save_images:
                plt.savefig(f'{folder_path}/electricity_price_week1_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the heat value
            fig, ax = plt.subplots()
            ax.plot(time_index, heat_value, label="Heat value [CHF/kWh]")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Heat Value [CHF/kWh]")
            ax.set_title("Heat Value")
            ax.legend()
            plt.xticks(rotation=45)
            ax.set_xlim([time_index[0], time_index[167]])
            if save_images:
                plt.savefig(f'{folder_path}/heat_value_week1_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the ambient temperature
            fig, ax = plt.subplots()
            ax.plot(time_index, T_amb, label="Ambient Temperature [°C]")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Temperature [°C]")
            ax.set_title("Ambient Temperature")
            ax.legend()
            plt.xticks(rotation=45)
            ax.set_xlim([time_index[0], time_index[167]])
            if save_images:
                plt.savefig(f'{folder_path}/ambient_temperature_week1_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the consumer temperature
            fig, ax = plt.subplots()
            ax.plot(time_index, T_cons1.value, label="Consumer 1 Temperature [°C]")
            ax.plot(time_index, T_cons2.value, label="Consumer 2 Temperature [°C]")
            ax.plot(time_index, T_cons3.value, label="Consumer 3 Temperature [°C]")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Temperature [°C]")
            ax.set_title("Target Consumer Temperature")
            ax.legend()
            plt.xticks(rotation=45)
            ax.set_xlim([time_index[0], time_index[167]])
            if save_images:
                plt.savefig(f'{folder_path}/consumer_temperature_week1_{name_version}.pdf', format="pdf", bbox_inches="tight")

            # Plot the heat demand and supply
            fig, ax = plt.subplots()
            ax.plot(time_index, heat_consumption, label="Heat Demand [kW]")
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Heat Demand [kW]")
            ax.set_title("Heat Demand")
            ax.legend()
            plt.xticks(rotation=45)
            ax.set_xlim([time_index[0], time_index[167]])
            if save_images:
                plt.savefig(f'{folder_path}/heat_demand_week1_{name_version}.pdf', format="pdf", bbox_inches="tight")
            
            ax.plot(heat_consumption, label="Heat Demand [kW]")
            ax.plot(time_index, heat_h2_used, label="Hydrogen Heat [kW]", linestyle='--')
            ax.plot(time_index, heat_hp_used, label="Heat Pump Heat [kW]", linestyle='-.')
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Heat [kW]")
            ax.set_title("Heat Demand and Supply")
            ax.legend()
            plt.xticks(rotation=45)
            ax.set_xlim([time_index[0], time_index[167]])
            if save_images:
                plt.savefig(f'{folder_path}/heat_demand_supply_week1_{name_version}.pdf', format="pdf", bbox_inches="tight")

            heat_gen_stacked = np.vstack((heat_el_used,heat_fc_used,heat_hp_used))

            fig, ax = plt.subplots()
            ax.stackplot(time_index, heat_gen_stacked, labels=["Electrolyser","Fuel cell","Heat Pump"])
            ax.plot(time_index, heat_consumption, label="Heat Demand", color='black', linestyle='--')
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Heat [kW]")
            ax.set_title("Heat Demand and Supply")
            ax.legend()
            plt.xticks(rotation=45)
            ax.set_xlim([time_index[0], time_index[167]])
            if save_images:
                plt.savefig(f'{folder_path}/heat_demand_supply_stack_week1_{name_version}.pdf', format="pdf", bbox_inches="tight")

            if plot_results:
                plt.show()

if __name__ == "__main__":
    main()
