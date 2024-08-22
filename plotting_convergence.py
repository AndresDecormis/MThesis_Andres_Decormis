import os
import pandas as pd
import matplotlib.pyplot as plt
import additional_functions as af
from cycler import cycler
import parametersv1 as p

#------
# Configuration of this run
af.configure_plots(style='fancy', colors='diverging')
# Name of the version
centralised_cost = 228.052         # Centralised total value TODO: Change this value for each run
centralised_heat_value = 0.02569    # Centralised heat value TODO: Change this value for each run
version = "decentralisedv5-bkw_green-high_inertia" # Name of the folder with the decentralised results
start_iteration = 10
save_figures = True
plot_figures = False
# color_list = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC', '#E6E6E6'] # Black and grey colors
# color_list = ["#d7e1ee", "#bfcbdb", "#a4a2a8", "#df8879", "#c86558", "#b04238"] # Blue and red colors
color_list = ["#115f9a", "#1984c5", "#22a7f0", "#48b5c4", "#76c68f", "#a6d75b"] # Blue and green colors
totalcycle = cycler(color= color_list, marker=['o', '+', 'x', '*', '.', 'X'])
#------

# Define the path results folder
data_folder = f"results/{version}"

# Open each folder in this directory and get the convergence data
primal_gap = pd.DataFrame()
cost_value = pd.DataFrame()
heat_value = pd.DataFrame()

# Find the right order of the folders
list_folders = os.listdir(data_folder)
# Just use folders, not other files
list_folders = [folder for folder in list_folders if os.path.isdir(os.path.join(data_folder, folder))]
list_folders.sort(key=lambda x: float(x.replace("rho_", "").replace("_", ".")))
print(list_folders)

for rho_folder in list_folders:
    rho_name    = rho_folder.replace("rho_", "")
    rho_value   = float(rho_name.replace("_", "."))
    print(f"Processing rho = {rho_value}")
    # Get the convergence data from the CSV file
    data_file   = os.path.join(data_folder, rho_folder, "convergence.csv")
    data        = pd.read_csv(data_file)
    # Get the values and copy them to the dataframes with the rho value in the column name
    primal_gap[rho_name] = data["PrimalGap"]    
    cost_value[rho_name] = data["Cost"]
    heat_value[rho_name] = data["HeatValue"]

# Making a list with header names using . instead of _ for the rho values
header_names = [name.replace("_", ".") for name in primal_gap.columns]
print(header_names)

# Plot the primal gap for each rho value - use ax, fig to plot multiple lines on the same plot
fig, ax = plt.subplots()
ax.set_prop_cycle(totalcycle)
ax.plot(primal_gap[start_iteration:], label=header_names)
ax.axhline(y=0, color='r', linestyle='--', label="Optimal Gap")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"Primal Gap, $||\mathbf{r}||_1$")
ax.set_title("Primal Gap Convergence")
plt.legend(title=r"$\rho$")
if save_figures:
    plt.savefig(f"results/{version}/primal_gap_convergence.pdf", format="pdf", bbox_inches="tight")

# Plot the cost value for each rho value - use ax, fig to plot multiple lines on the same plot
fig, ax = plt.subplots()
ax.set_prop_cycle(totalcycle)
ax.plot(cost_value[start_iteration:], label=header_names)
# Plot the centralised cost value
ax.axhline(y=centralised_cost, color='r', linestyle='--', label="Optimal Cost")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"Total Cost Value, $\mathcal{J}_{\mathrm{total}}$ [CHF]")
ax.set_title("Cost Value Convergence")
plt.legend(title=r"$\rho$")
if save_figures:
    plt.savefig(f"results/{version}/cost_value_convergence.pdf", format="pdf", bbox_inches="tight")

# Plot the heat value for each rho value - use ax, fig to plot multiple lines on the same plot
fig, ax = plt.subplots()
ax.set_prop_cycle(totalcycle)
ax.plot(heat_value[start_iteration:]*p.T**-1, label=header_names)
# Plot the centralised heat value
ax.axhline(y=centralised_heat_value, color='r', linestyle='--', label="Optimal Heat Value")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"Avg Heat Value, $\bar{\lambda}$ [CHF/kWh]")
ax.set_title("Heat Value Convergence")
plt.legend(title=r"$\rho$")
if save_figures:
    plt.savefig(f"results/{version}/heat_value_convergence.pdf", format="pdf", bbox_inches="tight")

if plot_figures:
    plt.plot()