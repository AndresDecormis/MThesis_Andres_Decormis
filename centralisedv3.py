# Description: Centralised optimisation of the energy system
# Features: simple modelling of components, uses relaxed temperature constraint

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parametersv1 import *

# Save results condition
save_results = False
plot_results = True
save_images  = False
name_version = "centralisedv3-standard"


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

# Import data of boundary conditions
data        = pd.read_csv("data/data2023.csv")          # Import data
data["Electricity_priceCHF"] = data["Electricity_priceCHF"].clip(lower=0.1) # Clip negative prices
data["Electricity_priceCHF"] = 0.001 * data["Electricity_priceCHF"] # Convert to CHF/kWh
price_el    = data["Electricity_priceCHF"].to_numpy()       # Electricity price [CHF/kWh]
T_amb       = data["TemperatureC"].to_numpy()               # Ambient temperature [°C]
price_h2    = hydrogen_market_price * np.ones(T)            # Hydrogen price [CHF/kgH2] - 3.3: https://data.sccer-jasm.ch/import-prices/2020-08-01/
q_gain      = np.zeros(T)                                   # Heat gain from sun irradiation [kW]

T_max_cons1_relaxed = T_max_allowed(T_cons1_max, alpha_cons1)
T_max_cons2_relaxed = T_max_allowed(T_cons2_max, alpha_cons2)
T_max_cons3_relaxed = T_max_allowed(T_cons3_max, alpha_cons3)

# Define the decision variables
p_imp   = cp.Variable(T,nonneg=True) # Import electricity [kW]
p_exp   = cp.Variable(T,nonneg=True) # Export electricity [kW]
h2_imp  = cp.Variable(T,nonneg=True) # Import hydrogen [kg]
h2_exp  = cp.Variable(T,nonneg=True) # Export hydrogen [kg]

# Define the dependant variables
# Power
l_h2    = cp.Variable(T,nonneg=True) # load of hydrogen system [kW]
l_el    = cp.Variable(T,nonneg=True) # load of electrolyser [kW]
l_co    = cp.Variable(T,nonneg=True) # load of compressor [kW]
l_hp    = cp.Variable(T,nonneg=True) # load of heat pump [kW]
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


# Define the constraints
# Define node balance constraints
network_balance = [p_imp    == l_h2 + l_hp,
                   q_cons1 + q_cons2 + q_cons3 - (q_h2 + q_hp) == 0,
                   l_h2     == l_el + l_co,
                   q_h2     == q_el + q_fc,
                   q_gen_el == q_el + q_was_el,
                   q_gen_fc == q_fc + q_was_fc,
                   p_exp    == p_fc]
# Define heat pump constraints
heat_pump = [q_hp   <= q_hp_max,
             q_hp   >= 0,
             q_hp   == COP_hp * l_hp]
# Define electrolyser constraints
electrolyser = [l_el        <= l_el_max,
                l_el        >= 0,
                h2_prod     == eff_el_h2 * l_el / HHV_H2,
                q_gen_el    == (1 - eff_el_h2) * eff_el_th * l_el]
# Define fuel cell constraints
fuel_cell = [p_fc       <= p_fc_max,
             p_fc       >= 0,
             h2_fc      == p_fc / (eff_fc_h2 * HHV_H2),
             q_gen_fc   == (1 - eff_fc_h2) * eff_fc_th * p_fc]
# Define compressor constraints
compressor = [l_co      <= l_co_max,
              l_co      >= 0,
              l_co      == k_co * h2_prod]
# Define storage constraints
storage = [h2_sto[0]        == h2_sto_max/2,
           h2_sto[T-1]      == h2_sto_max/2,
           h2_sto           <= h2_sto_max,
           h2_sto           >= 0,
           h2_imp           <= h2_imp_max,
           h2_exp           <= h2_exp_max,
           h2_sto[1:T]      == h2_sto[0:T-1] + h2_sto_eff * h2_prod[0:T-1] - (1/h2_sto_eff) * h2_fc[0:T-1] + h2_sto_eff * h2_imp[0:T-1] - (1/h2_sto_eff) * h2_exp[0:T-1]]
# Define heat consumer heat constraints
consumer = [T_cons1         >= T_cons1_min,
            T_cons1         <= T_max_cons1_relaxed,
            T_cons1[0]      == (T_cons1_max + T_cons1_min) / 2,
            T_cons1[1:T]    == alpha_cons1 * T_cons1[0:T-1] + (1-alpha_cons1) * (T_amb[0:T-1] + R_cons1 * (q_cons1[0:T-1] + q_gain[0:T-1])),
            T_cons2         >= T_cons2_min,
            T_cons2         <= T_max_cons2_relaxed,
            T_cons2[0]      == (T_cons2_max + T_cons2_min) / 2,
            T_cons2[1:T]    == alpha_cons2 * T_cons2[0:T-1] + (1-alpha_cons2) * (T_amb[0:T-1] + R_cons2 * (q_cons2[0:T-1] + q_gain[0:T-1])),
            T_cons3         >= T_cons3_min,
            T_cons3         <= T_max_cons3_relaxed,
            T_cons3[0]      == (T_cons3_max + T_cons3_min) / 2,
            T_cons3[1:T]    == alpha_cons3 * T_cons3[0:T-1] + (1-alpha_cons3) * (T_amb[0:T-1] + R_cons3 * (q_cons3[0:T-1] + q_gain[0:T-1]))]

# Define the objective function
objective   = cp.Minimize(price_el.T @ p_imp - price_el.T @ p_exp + price_h2.T @ h2_imp - price_h2.T @ h2_exp)
# Define the constraints
constraints = network_balance + heat_pump + electrolyser + fuel_cell + compressor + storage + consumer
# Create the problem
problem     = cp.Problem(objective, constraints)
# Solve the problem
problem.solve(reoptimize=True,solver=cp.GUROBI, verbose=True)

# Processing of results
# Main results
total_cost          = problem.value                 # scalar 
heat_value          = constraints[1].dual_value     # vector
average_heat_value  = np.average(heat_value)        # scalar
total_expenses      = price_el.T @ p_imp.value + price_h2.T @ h2_imp.value # scalar
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


# Print the main results
print("Minimum cost: ", total_cost)
print("Total expenses: ", total_expenses)
print("Expenses from electricity: ", price_el.T @ p_imp.value)
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

if save_results:
    # Save the results to a text file
    text_output = open(f"text-outputs/{name_version}.txt", "w")
    text_output.write("Minimum cost: " + str(total_cost) + " CHF \n")
    text_output.write("Total expenses: " + str(total_expenses) + " CHF \n")
    text_output.write("Expenses from electricity: " + str(price_el.T @ p_imp.value) + " CHF \n")
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

# For more detailed diagnostics:
print(problem.status)  # Check the status of the solution

if save_results:
    # Save the results
    # Main results
    results = pd.DataFrame({"Time": np.arange(0,T,1),
                            "HeatValue": np.round(heat_value,5),
                            "HeatH2": np.round(q_h2.value,5),
                            "HeatEl": np.round(q_el.value,5),
                            "HeatFC": np.round(q_fc.value,5),
                            "HeatHP": np.round(q_hp.value,5),
                            "HeatConsumption": np.round(heat_consumption,5),
                            "HeatSupply": np.round(heat_supply,5),
                            "HeatWaste": np.round(heat_waste,5),
                            "ElectricityDemand": np.round(electricity_demand,5),
                            "H2SystemLoad": np.round(l_h2.value,5),
                            "ElectrolyserLoad": np.round(l_el.value,5),
                            "CompressorLoad": np.round(l_co.value,5),
                            "HeatPumpLoad": np.round(l_hp.value,5),
                            "ElectricitySupply": np.round(electricity_supply,5),
                            "FuelCellGeneration": np.round(p_fc.value,5),
                            "NetElectricity": np.round(electricity_demand - electricity_supply,5),
                            "H2Produced": np.round(h2_prod.value,5),
                            "H2Used": np.round(h2_fc.value,5),
                            "H2Stored": np.round(h2_sto.value,5),
                            "H2Imported": np.round(h2_imp.value,5),
                            "H2Exported": np.round(h2_exp.value,5),
                            "TCons1": np.round(T_cons1.value,5),
                            "TCons2": np.round(T_cons2.value,5),
                            "TCons3": np.round(T_cons3.value,5),
                            "TConsAvg": np.round((T_cons1.value + T_cons2.value + T_cons3.value) / 3,5),})
    # Save to csv
    results.to_csv(f"results/{name_version}.csv", index=False)



if plot_results:
    # Plot the results
    # Yearly results
    # Plot the power import
    plt.figure()
    plt.plot(p_imp.value, label="Import electricity [kW]")
    plt.xlabel("Time [h]")
    plt.ylabel("Power [kW]")
    plt.title("Power Import")
    plt.legend()
    # save figure
    if save_images:
        plt.savefig(f'results/figures/{name_version}/power_import_{name_version}.png')

    # Plot the power export
    plt.figure()
    plt.plot(p_exp.value, label="Export electricity [kW]")
    plt.xlabel("Time [h]")
    plt.ylabel("Power [kW]")
    plt.title("Power Export")
    plt.legend()
    
    if save_images:
        # save figure
        plt.savefig(f'power_export_{name_version}.png')

    # Plot the electricity price
    plt.figure()
    plt.plot(price_el, label="Electricity Price [CHF/kWh]")
    plt.xlabel("Time [h]")
    plt.ylabel("Price [CHF/kWh]")
    plt.title("Electricity Price")
    plt.legend()
    if save_images:
        # save figure
        plt.savefig(f'results/figures/{name_version}/electricity_price_{name_version}.png')

    # Plot the ambient temperature
    plt.figure()
    plt.plot(T_amb, label="Ambient Temperature [°C]")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [°C]")
    plt.title("Ambient Temperature")
    plt.legend()
    if save_images:
        # save figure
        plt.savefig(f'results/figures/{name_version}/ambient_temperature_{name_version}.png')

    # Plot the consumer temperature
    plt.figure()
    plt.plot(T_cons1.value, label="Consumer 1 Temperature [°C]")
    plt.plot(T_cons2.value, label="Consumer 2 Temperature [°C]")
    plt.plot(T_cons3.value, label="Consumer 3 Temperature [°C]")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [°C]")
    plt.title("Real Consumer Temperature")
    plt.legend()
    if save_images:
        # save figure
        plt.savefig(f'results/figures/{name_version}/consumer_temperature_{name_version}.png')

    # Plot the excess temperature
    plt.figure()
    plt.plot(np.maximum(0,T_cons1.value - T_cons1_max), label="Consumer 1 Excess Temperature [°C]")
    plt.plot(np.maximum(0,T_cons2.value - T_cons2_max), label="Consumer 2 Excess Temperature [°C]")
    plt.plot(np.maximum(0,T_cons3.value - T_cons3_max), label="Consumer 3 Excess Temperature [°C]")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [°C]")
    plt.title("Excess Temperature")
    plt.legend()
    if save_images:
        # save figure
        plt.savefig(f'results/figures/{name_version}/excess_temperature_{name_version}.png')

    # Plot the heat demand
    plt.figure()
    plt.plot(q_cons1.value, label="Consumer 1 Heat Demand [kW]")
    plt.plot(q_cons2.value, label="Consumer 2 Heat Demand [kW]")
    plt.plot(q_cons3.value, label="Consumer 3 Heat Demand [kW]")
    plt.xlabel("Time [h]")
    plt.ylabel("Heat Demand [kW]")
    plt.title("Heat Demand")
    plt.legend()
    if save_images:
        # save figure
        plt.savefig(f'results/figures/{name_version}/heat_demand_{name_version}.png')

    # Plot the heat value
    plt.figure()
    plt.plot(constraints[1].dual_value, label="Heat value [CHF/kWh]")
    plt.xlabel("Time [h]")
    plt.ylabel("Heat Value [CHF/kWh]")
    plt.title("Heat Value")
    plt.legend()
    if save_images:
        # save figure
        plt.savefig(f'results/figures/{name_version}/heat_value_{name_version}.png')

    # Plot the hydrogen storage
    plt.figure()
    plt.plot(h2_sto.value, label="Hydrogen Storage [kg]")
    plt.xlabel("Time [h]")
    plt.ylabel("Storage [kg]")
    plt.title("Hydrogen Storage")
    plt.legend()
    if save_images:
        plt.savefig(f'results/figures/{name_version}/hydrogen_storage_{name_version}.png')

    # One week results
    plt.figure()
    plt.plot(p_imp.value[0:168], label="Import electricity [kW]")
    plt.plot(p_exp.value[0:168], label="Export electricity [kW]")
    plt.xlabel("Time [h]")
    plt.ylabel("Power [kW]")
    plt.title("Power Import and Export")
    plt.legend()
    if save_images:
        plt.savefig(f'results/figures/{name_version}/power_import_export_week1_{name_version}.png')

    # Plot the electricity price
    plt.figure()
    plt.plot(price_el[0:168], label="Electricity Price [CHF/kWh]")
    plt.xlabel("Time [h]")
    plt.ylabel("Price [CHF/kWh]")
    plt.title("Electricity Price")
    plt.legend()
    if save_images:
        plt.savefig(f'results/figures/{name_version}/electricity_price_week1_{name_version}.png')

    # Plot the heat value
    plt.figure()
    plt.plot(constraints[1].dual_value[0:168], label="Heat value [CHF/kWh]")
    plt.xlabel("Time [h]")
    plt.ylabel("Heat Value [CHF/kWh]")
    plt.title("Heat Value")
    plt.legend()
    if save_images:
        plt.savefig(f'results/figures/{name_version}/heat_value_week1_{name_version}.png')

    # Plot the ambient temperature
    plt.figure()
    plt.plot(T_amb[0:168], label="Ambient Temperature [°C]")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [°C]")
    plt.title("Ambient Temperature")
    plt.legend()
    if save_images:
        plt.savefig(f'results/figures/{name_version}/ambient_temperature_week1_{name_version}.png')

    # Plot the consumer temperature
    plt.figure()
    plt.plot(T_cons1.value[0:168], label="Consumer 1 Temperature [°C]")
    plt.plot(T_cons2.value[0:168], label="Consumer 2 Temperature [°C]")
    plt.plot(T_cons3.value[0:168], label="Consumer 3 Temperature [°C]")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [°C]")
    plt.title("Target Consumer Temperature")
    plt.legend()
    if save_images:
        plt.savefig(f'results/figures/{name_version}/consumer_temperature_week1_{name_version}.png')

    # Plot the heat demand and supply
    plt.figure()
    plt.plot(heat_consumption[0:168], label="Heat Demand [kW]")
    plt.plot(q_h2.value[0:168], label="Hydrogen Heat [kW]", linestyle='--')
    plt.plot(q_hp.value[0:168], label="Heat Pump Heat [kW]", linestyle='-.')
    plt.xlabel("Time [h]")
    plt.ylabel("Heat [kW]")
    plt.title("Heat Demand and Supply")
    plt.legend()
    if save_images:
        plt.savefig(f'results/figures/{name_version}/heat_demand_supply_week1_{name_version}.png')

    y = np.vstack((q_h2.value[0:168],q_hp.value[0:168]))
    x = np.arange(0,168,1)

    plt.figure()
    plt.stackplot(x,y, labels=["Hydrogen","Heat Pump"])
    plt.plot(heat_consumption[0:168], label="Heat Demand")
    plt.xlabel("Time [h]")
    plt.ylabel("Heat [kW]")
    plt.title("Heat Demand and Supply")
    plt.legend()
    if save_images:
        plt.savefig(f'results/figures/{name_version}/heat_demand_supply_stack_week1_{name_version}.png')

    # Plot the max allowable temperature
    plt.figure()
    plt.plot(T_max_cons1_relaxed, label="Consumer 1 Max Temperature [°C]")
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [°C]")
    plt.title("Max Allowable Temperature")
    plt.legend()
    if save_images:
        plt.savefig(f'results/figures/{name_version}/max_temperature_{name_version}.png')

    plt.show()