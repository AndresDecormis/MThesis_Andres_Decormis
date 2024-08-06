import os
import pandas as pd
import matplotlib.pyplot as plt
import additional_functions as af

#------
# Configuration of this run
af.configure_plots(style='fancy', colors='grayscale')
# Name of the version
version = "decentralisedv5-comparison" 
start_iteration = 10
save_figures = True
#------

# Define the path results folder
data_folder = f"results/{version}"

# Open each folder in this directory and get the convergence data
primal_gap = pd.DataFrame()
cost_value = pd.DataFrame()
heat_value = pd.DataFrame()

# Find the right order of the folders
list_folders = os.listdir(data_folder)
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
ax.plot(primal_gap[start_iteration:])
ax.set_xlabel("Iteration")
ax.set_ylabel("Primal Gap")
ax.set_title("Primal Gap Convergence")
plt.legend(header_names, title=r"$\rho$ parameter")
if save_figures:
    plt.savefig(f"results/{version}/primal_gap_convergence.pdf", format="pdf", bbox_inches="tight")

# Plot the cost value for each rho value - use ax, fig to plot multiple lines on the same plot
fig, ax = plt.subplots()
ax.plot(cost_value[start_iteration:])
ax.set_xlabel("Iteration")
ax.set_ylabel("Cost Value")
ax.set_title("Cost Value Convergence")
plt.legend(header_names, title=r"$\rho$ parameter")
if save_figures:
    plt.savefig(f"results/{version}/cost_value_convergence.pdf", format="pdf", bbox_inches="tight")

# Plot the heat value for each rho value - use ax, fig to plot multiple lines on the same plot
fig, ax = plt.subplots()
ax.plot(heat_value[start_iteration:])
ax.set_xlabel("Iteration")
ax.set_ylabel("Heat Value")
ax.set_title("Heat Value Convergence")
plt.legend(header_names, title=r"$\rho$ parameter")
if save_figures:
    plt.savefig(f"results/{version}/heat_value_convergence.pdf", format="pdf", bbox_inches="tight")

plt.plot()