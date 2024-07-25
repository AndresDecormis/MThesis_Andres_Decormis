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

# Configure the plotting
plot_results = False
folder_name  = "decentralisedv4"


rho_values     = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]     # Initial value of the penalty parameter

# Define the ADMM algorithm parameters
max_iter        = 50   # Maximum number of iterations
tolerance_primal= 1e-4 # Tolerance of the primal residual
tolerance_dual  = 1e-4 # Tolerance of the dual residual
n_agents        = 5         # Number of agents

# Import data of boundary conditions
data        = pd.read_csv("data/data2023.csv")          # Import data
data["Electricity_priceCHF"] = data["Electricity_priceCHF"].clip(lower=0.1) # Clip negative prices
data["Electricity_priceCHF"] = 0.001 * data["Electricity_priceCHF"] # Convert to CHF/kWh
price_el    = data["Electricity_priceCHF"].to_numpy()       # Electricity price [CHF/kWh]
T_amb       = data["TemperatureC"].to_numpy()               # Ambient temperature [°C]
price_h2    = hydrogen_market_price * np.ones(T)            # Hydrogen price [CHF/kgH2] - 3.3: https://data.sccer-jasm.ch/import-prices/2020-08-01/
q_gain      = np.zeros(T)                                   # Heat gain from sun irradiation [kW]

for initial_rho in rho_values:
    rho     = cp.Parameter(nonneg=True) # Penalty parameter
    # Define the decision variables
    l_h2    = cp.Variable(T,nonneg=True) # load of hydrogen system [kW]
    l_hp    = cp.Variable(T,nonneg=True) # load of heat pump [kW]
    p_exp   = cp.Variable(T,nonneg=True) # Export electricity [kW]
    h2_imp  = cp.Variable(T,nonneg=True) # Import hydrogen [kg]
    h2_exp  = cp.Variable(T,nonneg=True) # Export hydrogen [kg]

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

    # Define fixed and lagrangian variables
    lambda_fixed            = cp.Parameter(T)
    q_h2_fixed              = cp.Parameter(T)
    q_hp_fixed              = cp.Parameter(T)
    q_cons1_fixed           = cp.Parameter(T)
    q_cons2_fixed           = cp.Parameter(T)
    q_cons3_fixed           = cp.Parameter(T)
    q_average               = cp.Parameter(T)

    # Define functions to build optimisation problems`
    # Hydrogen system
    def hydrogen_system_optimisation():
        """
        hydrogen_system_optimisation problem
        
        """
        # Define the cost function
        cost_h2    = price_el.T @ l_h2 - price_el.T @ p_exp + price_h2.T @ h2_imp - price_h2.T @ h2_exp

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
        augm_lagr_h2 = cost_h2 - lambda_fixed.T @ q_h2 + rho/2 * cp.sum_squares(n_agents * q_average + q_h2_fixed - q_h2)
        constraints_h2 = hydrogen_network_balance + electrolyser + fuel_cell + compressor + storage
        objective_h2 = cp.Minimize(augm_lagr_h2)
        # Create the problem
        problem_h2 = cp.Problem(objective_h2, constraints_h2)
        return problem_h2

    # Heat pump
    def heat_pump_optimisation():
        """
        heat_pump_optimisation obtains the optimal heat produced by the hydrogen system
        
        """
        # Define the cost function
        cost_hp    = price_el.T @ l_hp

    # Define heat pump constraints
        heat_pump = [q_hp <= q_hp_max,
                    q_hp >= 0,
                    q_hp == COP_hp * l_hp]
        # Define augmented lagrangian
        augm_lagr_hp = cost_hp - lambda_fixed.T @ q_hp + rho/2 * cp.sum_squares(n_agents * q_average + q_hp_fixed - q_hp)

        constraints_hp = heat_pump
        objective_hp = cp.Minimize(augm_lagr_hp)

        # Create the problem
        problem_hp = cp.Problem(objective_hp, constraints_hp)
        return problem_hp
    
    # Define the temperature constraints
    def T_max_allowed(T_max_ideal: int, alpha_consumer: int):
        """
        Function to calculate the maximum allowed temperature for the consumer
        """
        # Define cost
        cost_excess = 1e12 * np.ones(T) # Cost of excess temperature [CHF/°C]
        # Define the decision variables
        T_consumer = cp.Variable(T) # Consumer temperature [°C]
        # Define the constraints
        constraints_temperature = [T_consumer >= T_max_ideal,
                                   T_consumer[1:T] >= alpha_consumer * T_consumer[0:T-1] + (1-alpha_consumer) * (T_amb[0:T-1])]
        # Define the objective function
        objective_temperature = cp.Minimize(cost_excess.T @ T_consumer)
        # Create the problem
        problem_temperature = cp.Problem(objective_temperature, constraints_temperature)
        # Solve the problem
        problem_temperature.solve(reoptimize=True,solver=cp.GUROBI, verbose=True)
        # Obtain the relaxed maximum temperature
        T_max_relaxed = T_consumer.value
        return T_max_relaxed

    # Heat consumer 1
    def consumer_1_optimisation():
        """
        consumer_1_optimisation problem
        
        """
        # Define heat consumer heat constraints

        T_max_cons1_relaxed = T_max_allowed(T_cons1_max, alpha_cons1)
        consumer_1 = [T_cons1       <= T_max_cons1_relaxed,
                      T_cons1       >= T_cons1_min,
                      T_cons1[0]    == (T_cons1_max + T_cons1_min)/2,
                      T_cons1[1:T]  == alpha_cons1 * T_cons1[0:T-1] + (1-alpha_cons1) * (T_amb[0:T-1] + R_cons1 * (q_cons1[0:T-1] + q_gain[0:T-1]))]
        # Define augmented lagrangian
        augm_lagr_cons1 = lambda_fixed.T @ q_cons1 + rho/2 * cp.sum_squares(n_agents * q_average - q_cons1_fixed + q_cons1)

        constraints_cons1 = consumer_1
        objective_cons1 = cp.Minimize(augm_lagr_cons1)

        # Create the problem
        problem_cons1 = cp.Problem(objective_cons1, constraints_cons1)
        return problem_cons1

    # Heat consumer 2
    def consumer_2_optimisation():
        """
        consumer_2_optimisation problem
        
        """
        # Define heat consumer heat constraints

        T_max_cons2_relaxed = T_max_allowed(T_cons2_max, alpha_cons2)
        consumer_2 = [T_cons2       <= T_max_cons2_relaxed,
                      T_cons2       >= T_cons2_min,
                      T_cons2[0]    == (T_cons2_max + T_cons2_min)/2,
                      T_cons2[1:T]  == alpha_cons2 * T_cons2[0:T-1] + (1-alpha_cons2) * (T_amb[0:T-1] + R_cons2 * (q_cons2[0:T-1] + q_gain[0:T-1]))]
        # Define augmented lagrangian
        augm_lagr_cons2 = lambda_fixed.T @ q_cons2 + rho/2 * cp.sum_squares(n_agents * q_average - q_cons2_fixed + q_cons2)

        constraints_cons2 = consumer_2
        objective_cons2 = cp.Minimize(augm_lagr_cons2)

        # Create the problem
        problem_cons2 = cp.Problem(objective_cons2, constraints_cons2)
        return problem_cons2

    # Heat consumer 3
    def consumer_3_optimisation():
        """
        consumer_3_optimisation problem
        
        """
        # Define heat consumer heat constraints

        T_max_cons3_relaxed = T_max_allowed(T_cons3_max, alpha_cons3)
        consumer_3 = [T_cons3       <= T_max_cons3_relaxed,
                      T_cons3       >= T_cons3_min,
                      T_cons3[0]    == (T_cons3_max + T_cons3_min)/2,
                      T_cons3[1:T]  == alpha_cons3 * T_cons3[0:T-1] + (1-alpha_cons3) * (T_amb[0:T-1] + R_cons3 * (q_cons3[0:T-1] + q_gain[0:T-1]))]
        # Define augmented lagrangian
        augm_lagr_cons3 = lambda_fixed.T @ q_cons3 + rho/2 * cp.sum_squares(n_agents * q_average - q_cons3_fixed + q_cons3)
        constraints_cons3 = consumer_3
        objective_cons3 = cp.Minimize(augm_lagr_cons3)
        # Create the problem
        problem_cons3 = cp.Problem(objective_cons3, constraints_cons3)
        return problem_cons3

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

    # Initial values
    lambda_fixed.value            = np.zeros(T) # Lagrangian variable [CHF/kWh]
    q_h2_fixed.value              = np.zeros(T) # Initial value of the hydrogen system heat
    q_hp_fixed.value              = np.zeros(T) # Initial value of the heat pump heat
    q_cons1_fixed.value           = np.zeros(T) # Initial value of the consumer 1 heat
    q_cons2_fixed.value           = np.zeros(T) # Initial value of the consumer 2 heat
    q_cons3_fixed.value           = np.zeros(T) # Initial value of the consumer 3 heat
    rho.value                     = initial_rho # Initial value of the penalty parameter
    q_average.value               = np.zeros(T) # Average heat value

    # Build the optimisation problems
    problem_h2      = hydrogen_system_optimisation()
    problem_hp      = heat_pump_optimisation()
    problem_cons1   = consumer_1_optimisation()
    problem_cons2   = consumer_2_optimisation()
    problem_cons3   = consumer_3_optimisation()


    # Storing heat values
    lambda_storage      = np.zeros((T,max_iter))
    primal_gap_storage  = np.zeros(max_iter)
    cost_storage        = np.zeros(max_iter)

    for iter in range(max_iter):

        q_average.value = (q_cons1_fixed.value + q_cons2_fixed.value + q_cons3_fixed.value - q_h2_fixed.value - q_hp_fixed.value) / n_agents

        # Update the hydrogen system
        q_h2.value          = hydrogen_system_optimal_value(problem_h2)
        q_h2_fixed.value    = q_h2.value

        q_average.value = (q_cons1_fixed.value + q_cons2_fixed.value + q_cons3_fixed.value - q_h2_fixed.value - q_hp_fixed.value) / n_agents

        # Update the heat pump
        q_hp.value          = heat_pump_optimal_value(problem_hp)
        q_hp_fixed.value    = q_hp.value

        q_average.value = (q_cons1_fixed.value + q_cons2_fixed.value + q_cons3_fixed.value - q_h2_fixed.value - q_hp_fixed.value) / n_agents

        # Update the heat consumer 1
        q_cons1.value       = consumer_1_optimal_value(problem_cons1)
        q_cons1_fixed.value = q_cons1.value

        q_average.value = (q_cons1_fixed.value + q_cons2_fixed.value + q_cons3_fixed.value - q_h2_fixed.value - q_hp_fixed.value) / n_agents

        # Update the heat consumer 2
        q_cons2.value       = consumer_2_optimal_value(problem_cons2)
        q_cons2_fixed.value = q_cons2.value

        q_average.value = (q_cons1_fixed.value + q_cons2_fixed.value + q_cons3_fixed.value - q_h2_fixed.value - q_hp_fixed.value) / n_agents

        # Update the heat consumer 3
        q_cons3.value       = consumer_3_optimal_value(problem_cons3)
        q_cons3_fixed.value = q_cons3.value

        q_average.value = (q_cons1_fixed.value + q_cons2_fixed.value + q_cons3_fixed.value - q_h2_fixed.value - q_hp_fixed.value) / n_agents

        # Residual calculation
        primal_residual = q_cons1.value + q_cons2.value + q_cons3.value - (q_h2.value + q_hp.value)                                             # Difference between heat consumption and production at every timestep [kWh]
        total_cost      = price_el.T @ (l_h2.value + l_hp.value) - price_el.T @ p_exp.value + price_h2.T @ h2_imp.value - price_h2.T @ h2_exp.value # Total cost [CHF]

        # Storing values
        lambda_storage[:,iter]  = lambda_fixed.value
        primal_gap_storage[iter]= np.linalg.norm(primal_residual,ord = 1)
        cost_storage[iter]      = total_cost


        # Update the dual variable
        lambda_fixed.value      = lambda_fixed.value + rho.value * (primal_residual)
        dual_residual           = lambda_fixed.value - lambda_storage[:,iter]

        # Update penalty parameter
        # rho.value = 2 * rho.value

        # Check for convergence
        # if all(abs(primal_residual)) < tolerance_primal and all(abs(dual_residual)) < tolerance_dual:
        #     break

        print("Iteration: ", iter," finished.\n")
        print("Current total primal residual: ", sum(abs(primal_residual))," \n")
        print("Current total dual residual: ", sum(abs(dual_residual))," \n")
        print("Minimum cost value: ", total_cost)
        # plt.figure()
        # plt.plot(lambda_fixed.value)
        # plt.show()


    

    # Processing of results
    # Main results
    total_cost          = total_cost                    # scalar 
    heat_value          = lambda_fixed.value            # vector
    average_heat_value  = np.average(heat_value)        # scalar
    total_expenses      = price_el.T @ l_h2.value + price_el.T @ l_hp.value + price_h2.T @ h2_imp.value # scalar
    total_revenue       = price_el.T @ p_exp.value + price_h2.T @ h2_exp.value # scalar
    # Heat results
    heat_consumption    = q_cons1.value + q_cons2.value + q_cons3.value
    total_heat_demand   = np.sum(heat_consumption)
    heat_supply         = q_h2.value + q_hp.value
    heat_waste          = q_was_el.value + q_was_fc.value
    # Electricity results
    electricity_demand          = l_h2.value + l_hp.value   # vector
    electricity_supply          = p_fc.value                # vector
    total_electricity_demand    = np.sum(electricity_demand) # scalar
    net_electricity             = total_electricity_demand - np.sum(electricity_supply) # scalar
    # Hydrogen results
    hydrogen_consumption    = h2_fc.value
    hydrogen_production     = h2_prod.value
    total_hydrogen_used     = np.sum(hydrogen_consumption)
    total_hydrogen_produced = np.sum(hydrogen_production)
    hydrogen_imported       = h2_imp.value
    hydrogen_exported       = h2_exp.value

    rhoname = str(rho.value).replace(".","_")
    # Print the optimal value and solution
    print("Minimum cost: ", total_cost)
    print("Total expenses: ", total_expenses)
    print("Expenses from electricity: ", price_el.T @ l_h2.value + price_el.T @ l_hp.value)
    print("Expenses from hydrogen: ", price_h2.T @ h2_imp.value)
    print("Total revenue: ", total_revenue)
    print("Revenue from electricity: ", price_el.T @ p_exp.value)
    print("Revenue from hydrogen: ", price_h2.T @ h2_exp.value)
    print("Heat cost: ", heat_value)
    print("Average heat cost: ", average_heat_value)
    print("Total heat demand:" , total_heat_demand)
    print("Total heat used from hydrogen system: ", np.sum(q_h2.value))
    print("Total heat used from heat pump: ", np.sum(q_hp.value))
    print("Total electricity consumed: ", total_electricity_demand)
    print("Total electricity produced: ", np.sum(electricity_supply))

    text_output = open(f"text-outputs/{folder_name}_rho_{rhoname}.txt", "w")
    text_output.write("Minimum cost: " + str(total_cost) + " CHF \n")
    text_output.write("Total expenses: " + str(total_expenses) + " CHF \n")
    text_output.write("Expenses from electricity: " + str(price_el.T @ l_h2.value + price_el.T @ l_hp.value) + " CHF \n")
    text_output.write("Expenses from hydrogen: " + str(price_h2.T @ h2_imp.value) + " CHF \n")
    text_output.write("Total revenue: " + str(total_revenue) + " CHF \n")
    text_output.write("Revenue from electricity: " + str(price_el.T @ p_exp.value) + " CHF \n")
    text_output.write("Revenue from hydrogen: " + str(price_h2.T @ h2_exp.value) + " CHF \n")
    text_output.write("Average heat value: " + str(average_heat_value) + " CHF / kWh \n")
    text_output.write("Total heat demand: " + str(total_heat_demand) + " kWh \n")
    text_output.write("Total heat used from hydrogen system: " + str(np.sum(q_h2.value)) + " kWh \n")
    text_output.write("Total heat used from heat pump: " + str(np.sum(q_hp.value)) + " kWh \n")
    text_output.write("Total electricity consumed: " + str(total_electricity_demand) + " kWh \n")
    text_output.write("Total electricity produced: " + str(np.sum(electricity_supply)) + " kWh \n")
    text_output.write("Total hydrogen produced: " + str(total_hydrogen_produced) + " kg \n")
    text_output.write("Total hydrogen used: " + str(total_hydrogen_used) + " kg \n")
    text_output.close()

    # Save the results
    # Results over iterations
    results_iterations = pd.DataFrame({"Iteration": range(iter),
                                       "PrimalGap": primal_gap_storage[0:iter],
                                       "Cost": cost_storage[0:iter],
                                       "HeatValue": np.sum(lambda_storage, axis=0)[0:iter]})
    # Save to csv
    results_iterations.to_csv(f"results/{folder_name}/rho_{rhoname}_convergence.csv", index=False)

    # Final results
    results = pd.DataFrame({"Time": np.arange(0,T,1),
                            "HeatValue": heat_value,
                            "HeatH2": q_h2.value,
                            "HeatEl": q_el.value,
                            "HeatFC": q_fc.value,
                            "HeatHP": q_hp.value,
                            "HeatConsumption": heat_consumption,
                            "HeatSupply": heat_supply,
                            "HeatWaste": heat_waste,
                            "ElectricityDemand": electricity_demand,
                            "H2SystemLoad": l_h2.value,
                            "ElectrolyserLoad": l_el.value,
                            "CompressorLoad": l_co.value,
                            "HeatPumpLoad": l_hp.value,
                            "ElectricitySupply": electricity_supply,
                            "FuelCellGeneration": p_fc.value,
                            "NetElectricity": electricity_demand - electricity_supply,
                            "H2Produced": h2_prod.value,
                            "H2Used": h2_fc.value,
                            "H2Stored": h2_sto.value,
                            "H2Imported": h2_imp.value,
                            "H2Exported": h2_exp.value,
                            "TCons1": T_cons1.value,
                            "TCons2": T_cons2.value,
                            "TCons3": T_cons3.value,
                            "TConsAvg": (T_cons1.value + T_cons2.value + T_cons3.value) / 3,})
    # Save to csv
    results.to_csv(f"results/{folder_name}/rho_{rhoname}_final.csv", index=False)



if plot_results:
    # Plot the results
    # Plot the power import
    plt.figure()
    plt.plot(l_h2.value + l_hp.value, label="Import electricity [kW]")
    plt.xlabel("Time [h]")
    plt.ylabel("Power [kW]")
    plt.title("Power Import")
    plt.legend()

    # Plot the power export
    plt.figure()
    plt.plot(p_exp.value, label="Export electricity [kW]")
    plt.xlabel("Time [h]")
    plt.ylabel("Power [kW]")
    plt.title("Power Export")
    plt.legend()

    # Plot the electricity price
    plt.figure()
    plt.plot(price_el, label="Electricity Price [CHF/kWh]")
    plt.xlabel("Time [h]")
    plt.ylabel("Price [CHF/kWh]")
    plt.title("Electricity Price")
    plt.legend()

    # Plot the ambient temperature
    plt.figure()
    plt.plot(T_amb, label="Ambient Temperature [°C]")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [°C]")
    plt.title("Ambient Temperature")
    plt.legend()

    # Plot the consumer temperature
    plt.figure()
    plt.plot(T_cons1.value, label="Consumer 1 Temperature [°C]")
    plt.plot(T_cons2.value, label="Consumer 2 Temperature [°C]")
    plt.plot(T_cons3.value, label="Consumer 3 Temperature [°C]")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [°C]")
    plt.title("Consumer Temperature")
    plt.legend()

    # Plot the heat demand
    plt.figure()
    plt.plot(q_cons1.value, label="Consumer 1 Heat Demand [kW]")
    plt.plot(q_cons2.value, label="Consumer 2 Heat Demand [kW]")
    plt.plot(q_cons3.value, label="Consumer 3 Heat Demand [kW]")
    plt.xlabel("Time [h]")
    plt.ylabel("Heat Demand [kW]")
    plt.title("Heat Demand")
    plt.legend()

    # Plot the heat value
    plt.figure()
    plt.plot(lambda_fixed.value, label="Heat value [CHF/kWh]")
    plt.xlabel("Time [h]")
    plt.ylabel("Heat Value [CHF/kWh]")
    plt.title("Heat Value")
    plt.legend()
    plt.savefig(f"results/{folder_name}/figures/Heat_value_rho_{rho.value}.png")

    plt.show()