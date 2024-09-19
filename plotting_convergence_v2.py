import os
import pandas as pd
import matplotlib.pyplot as plt
import additional_functions as af
from cycler import cycler
from parametersv2 import *

def main():

    # -----------------------------------------------------------------------------------------------
    # Configuration of this run
    af.configure_plots(style='fancy', colors='diverging')
    # -----------------------------------------------------------------------------------------------
    # Name of the version
    version                 = "distributedv8-ADMM-iwb_power_small-Medium_TI-True_EC-True_PV-True_BES-True_TES-True_EL-True_FC-Medium_TF-True_H2-current_H2Price-1_PWA_1e6" # Name of the folder with the decentralised results
    centralised_cost        = 8198.16131591796         # Centralised total value TODO: Change this value for each run
    centralised_heat_value  = 0.09144306089433517    # Centralised heat value TODO: Change this value for each run
    centralised_elec_value  = 0.3005733964051300    # Centralised heat value TODO: Change this value for each run

    start_iteration = 20

    save_figures = True
    plot_figures = False
    # -----------------------------------------------------------------------------------------------
    # Color cycle for the plots
    # color_list = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC', '#E6E6E6'] # Black and grey colors
    # color_list = ["#d7e1ee", "#bfcbdb", "#a4a2a8", "#df8879", "#c86558", "#b04238"] # Blue and red colors
    # color_list = ["#115f9a", "#1984c5", "#22a7f0", "#48b5c4", "#76c68f", "#a6d75b"] # Blue and green colors
    color_list = [
                    "#115f9a",  # Dark blue
                    "#1984c5",  # Medium blue
                    "#22a7f0",  # Light blue
                    "#29b6f6",  # Sky blue
                    "#48b5c4",  # Teal
                    "#4fa5a6",  # Aqua green
                    "#53b7a8",  # Soft turquoise
                    "#5ea893",  # Teal green
                    "#76c68f",  # Soft green
                    "#87cf60",  # Lime green
                    "#8fd175",  # Soft lime green
                    "#a6d75b",  # Light green
                    ]
    totalcycle = cycler(color= color_list, marker=['o', '+', 'x', '*', '.', 'X', 'D', 's', 'd', '^', 'v', '<'])
    # -----------------------------------------------------------------------------------------------

    # Define the path results folder
    data_folder = f"results/{version}"

    # Open each folder in this directory and get the convergence data
    primal_gap          = pd.DataFrame()
    primal_gap_elec     = pd.DataFrame()
    cost_value          = pd.DataFrame()
    heat_value          = pd.DataFrame()
    elec_value          = pd.DataFrame()

    # Find the right order of the folders
    list_folders = os.listdir(data_folder)
    # Just use folders, not other files
    list_folders = [folder for folder in list_folders if os.path.isdir(os.path.join(data_folder, folder))]
    # Sort the folders by the rho value, need to first remove the rho_, then replace _ with . and then remove what comes after the U (the U included)
    list_folders.sort(key=lambda x: float(x.replace("rho_", "").replace("_", ".").split("U")[0]))
    # list_folders.sort(key=lambda x: float(x.replace("rho_", "").replace("_", ".")))
    print(list_folders)

    # Extract data from the CSV files of each folder
    for rho_folder in list_folders:
        rho_name    = rho_folder.replace("rho_", "")
        # rho_value   = float(rho_name.replace("_", "."))
        # print(f"Processing rho = {rho_value}")
        # Get the convergence data from the CSV file
        data_file   = os.path.join(data_folder, rho_folder, "convergence.csv")
        data        = pd.read_csv(data_file)
        # Get the values and copy them to the dataframes with the rho value in the column name
        primal_gap[rho_name]        = data["PrimalGap"] 
        primal_gap_elec[rho_name]   = data["PrimalGapElec"]   
        cost_value[rho_name]        = data["Cost"]
        heat_value[rho_name]        = data["HeatValue"]
        elec_value[rho_name]        = data["ElecValue"]

    # Making a list with header names using . instead of _ for the rho values
    header_names = [name.replace("_", ".") for name in primal_gap.columns]
    # Changin the U in the name of each string to ", " 
    header_names = [name.replace("U", ", ") for name in header_names]
    print(header_names)

    # Plot the primal gap for each rho value - use ax, fig to plot multiple lines on the same plot
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(primal_gap[start_iteration:], label=header_names)
    ax.axhline(y=0, color='r', linestyle='--', label="Optimal Gap")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Primal residual, $||\mathbf{r}^{\mathrm{H}}||_1$")
    ax.set_title("Primal Gap Convergence - Heat")
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/primal_gap_convergence.pdf", format="pdf", bbox_inches="tight")

    # Plot the primal gap electricity for each rho value - use ax, fig to plot multiple lines on the same plot
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(primal_gap_elec[start_iteration:], label=header_names)
    ax.axhline(y=0, color='r', linestyle='--', label="Optimal Gap")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Primal residual, $||\mathbf{r}^{\mathrm{E}}||_1$")
    ax.set_title("Primal Gap Convergence - Electricity")
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/primal_gap_elec_convergence.pdf", format="pdf", bbox_inches="tight")

    # Plot the cost value for each rho value - use ax, fig to plot multiple lines on the same plot
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(cost_value[start_iteration:], label=header_names)
    # Plot the centralised cost value
    ax.axhline(y=centralised_cost, color='r', linestyle='--', label="Optimal Cost")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Total Cost Value, $\mathcal{J}_{\mathrm{total}}$ [CHF]")
    ax.set_title("Cost Value Convergence")
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/cost_value_convergence.pdf", format="pdf", bbox_inches="tight")

    # Plot the heat value for each rho value - use ax, fig to plot multiple lines on the same plot
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(heat_value[start_iteration:]*T**-1, label=header_names)
    # Plot the centralised heat value
    ax.axhline(y=centralised_heat_value, color='r', linestyle='--', label="Optimal Heat Value")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Avg Heat Value, $\bar{\lambda}^{\mathrm{H}}$ [CHF/kWh]")
    ax.set_title("Heat Value Convergence")
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/heat_value_convergence.pdf", format="pdf", bbox_inches="tight")

    # Plot the electricity value for each rho value - use ax, fig to plot multiple lines on the same plot
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(elec_value[start_iteration:]/T, label=header_names)
    # Plot the centralised electricity value
    ax.axhline(y=centralised_elec_value, color='r', linestyle='--', label="Optimal Elec Value")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Avg Elec Value, $\bar{\lambda}^{\mathrm{E}}$ [CHF/kWh]")
    ax.set_title("Electricity Value Convergence")
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/elec_value_convergence.pdf", format="pdf", bbox_inches="tight")

    if plot_figures:
        plt.plot()


if __name__ == "__main__":
    main()  # Run the main function