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
    version                 = "distributedv8-iwb_power_small-MediumTI-MediumTF-TruePV-TrueBES-TrueTES-TrueEL-TrueFC-currentH2p-3_PWA-Complete-CurrentH2-DynamicRho" # Name of the folder with the decentralised results
    centralised_cost        = 5039.38         # Centralised total value TODO: Change this value for each run
    centralised_heat_value  = 0.058551    # Centralised heat value TODO: Change this value for each run
    centralised_elec_value  = 0.2838976    # Centralised heat value TODO: Change this value for each run
    centralised_weighted_heat_value = 0.077621561  # Centralised weighted heat value TODO: Change this value for each run
    centralised_weighted_elec_value = 0.28998167    # Centralised weighted electricity value TODO: Change this value for each run
    tolerance_primal= 10          # Tolerance of the primal residual [kWh/year]
    tolerance_dual  = 10          # Tolerance of the dual residual [kWh/year]
    

    start_iteration = 0

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
    # totalcycle = cycler(color= color_list, marker=['o', '+', 'x', '*', '.', 'X', 'D', 's', 'd', '^', 'v', '<'])#, linestyle=['-', '--', '-.', ':'])
    totalcycle = cycler(color= color_list,linestyle=['-', '--', '-.', ':','-', '--', '-.', ':','-', '--', '-.', ':'])#, marker=['o', '+', 'x', '*', '.', 'X', 'D', 's', 'd', '^', 'v', '<'])#, linestyle=['-', '--', '-.', ':'])

    # -----------------------------------------------------------------------------------------------

    # Define the path results folder
    data_folder = f"results/{version}"

    # Open each folder in this directory and get the convergence data
    primal_gap          = pd.DataFrame()
    primal_gap_elec     = pd.DataFrame()
    cost_value          = pd.DataFrame()
    heat_value          = pd.DataFrame()
    elec_value          = pd.DataFrame()
    w_heat_value        = pd.DataFrame()
    w_elec_value        = pd.DataFrame()
    dual_gap_norm2      = pd.DataFrame()
    dual_gap_norm2_elec = pd.DataFrame()
    absolute_deviation  = pd.DataFrame()
    abs_diff_w_h_value  = pd.DataFrame()
    abs_diff_w_e_value  = pd.DataFrame()
    percent_diff_cost   = pd.DataFrame()

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
        primal_gap[rho_name]        = data["Norm1PrimalResidualHeat"] 
        primal_gap_elec[rho_name]   = data["Norm1PrimalResidualElec"]   
        cost_value[rho_name]        = data["Cost"]
        heat_value[rho_name]        = data["HeatValue"]
        elec_value[rho_name]        = data["ElecValue"]
        w_heat_value[rho_name]      = data["WeightedHeatValue"]
        w_elec_value[rho_name]      = data["WeightedElecValue"]
        dual_gap_norm2[rho_name]    = data["Norm2DualResidualHeat"]
        dual_gap_norm2_elec[rho_name] = data["Norm2DualResidualElec"]
        absolute_deviation[rho_name] = np.abs(cost_value[rho_name] - centralised_cost)
        abs_diff_w_h_value[rho_name] = np.abs(w_heat_value[rho_name] - centralised_weighted_heat_value)
        abs_diff_w_e_value[rho_name] = np.abs(w_elec_value[rho_name] - centralised_weighted_elec_value)
        percent_diff_cost[rho_name]  = absolute_deviation[rho_name]/centralised_cost*100

    # Making a list with header names using . instead of _ for the rho values
    header_names = [name.replace("_", ".") for name in primal_gap.columns]
    # Changin the U in the name of each string to ", " 
    header_names = [name.replace("U", ", ") for name in header_names]
    print(header_names)

    # Plot the primal gap for each rho value - use ax, fig to plot multiple lines on the same plot
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(primal_gap[start_iteration:], label=header_names)
    ax.axhline(y=0, color='black', linestyle='--', label="Optimal")
    # Color the region where the primal gap is below tolerance
    ax.fill_between(primal_gap.index, -tolerance_primal, tolerance_primal, color='lightgrey', alpha=0.5, label = r"$< \epsilon^{\mathrm{pri}}$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Primal residual, $||\mathbf{r}^{\mathrm{H}}||_1$ [kWh/year]")
    ax.set_title("Primal Gap Convergence - Heat")
    ax.xaxis.get_major_locator().set_params(integer=True)
    # Start in x = start_iteration
    ax.set_xlim(left=start_iteration)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/primal_gap_convergence.pdf", format="pdf", bbox_inches="tight")

    # Plot the primal gap convergence for heat in log scale
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(primal_gap[start_iteration:], label=header_names)
    ax.set_yscale('log')
    # ax.axhline(y=0, color='black', linestyle='--', label="Optimal")
    # Color the region where the primal gap is below tolerance - use axhspan to color the region
    ax.axhspan(-tolerance_primal, tolerance_primal, color='lightgrey', alpha=0.5, label = r"$< \epsilon^{\mathrm{pri}}$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Primal residual heat, $||\mathbf{r}^{\mathrm{H}}||_1$ [kWh/year]")
    # ax.set_title("Primal Gap Convergence - Heat")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(left=start_iteration)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/primal_gap_convergence_log.pdf", format="pdf", bbox_inches="tight")

    # Plot the dual gap norm2 for each rho value - use ax, fig to plot multiple lines on the same plot
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(dual_gap_norm2[start_iteration:], label=header_names)
    ax.set_yscale('log')
    ax.axhline(y=0, color='black', linestyle='--', label="Optimal")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Dual residual heat, $||\mathbf{r}^{\mathrm{H}}||_2$")
    ax.set_title("Dual Gap Convergence - Heat")
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/dual_gap_convergence_log.pdf", format="pdf", bbox_inches="tight")


    # Plot the primal gap convergence for electricity in log scale
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(primal_gap_elec[start_iteration:], label=header_names)
    ax.set_yscale('log')
    # ax.axhline(y=0, color='black', linestyle='--', label="Optimal")
    # Color the region where the primal gap is below tolerance - use axhspan to color the region
    ax.axhspan(-tolerance_primal, tolerance_primal, color='lightgrey', alpha=0.5, label = r"$< \epsilon^{\mathrm{pri}}$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Primal residual electricity, $||\mathbf{r}^{\mathrm{E}}||_1$ [kWh/year]")
    # ax.set_title("Primal Gap Convergence - Electricity")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(left=start_iteration)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/primal_gap_elec_convergence_log.pdf", format="pdf", bbox_inches="tight")


    # Plot the primal gap electricity for each rho value - use ax, fig to plot multiple lines on the same plot
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(primal_gap_elec[start_iteration:], label=header_names)
    ax.axhline(y=0, color='r', linestyle='--', label="Optimal Gap")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Primal residual electricity, $||\mathbf{r}^{\mathrm{E}}||_1$ [kWh/year]")
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
    ax.axhline(y=centralised_cost, color='black', linestyle='--', label=r"${J}_{\mathrm{op,optimal}}$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Total Cost Value, ${J}_{\mathrm{op}}$ [CHF]")
    # ax.set_title("Cost Value Convergence")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(left=start_iteration)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/cost_value_convergence.pdf", format="pdf", bbox_inches="tight")

    # Plot the cost value for each rho value - log scale
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(cost_value[start_iteration:], label=header_names)
    ax.set_yscale('log')
    # Plot the centralised cost value
    ax.axhline(y=centralised_cost, color='black', linestyle='--', label=r"${J}_{\mathrm{op,optimal}}$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Total Cost Value, ${J}_{\mathrm{op}}$ [CHF/year]")
    # ax.set_title("Cost Value Convergence")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(left=start_iteration)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/cost_value_convergence_log.pdf", format="pdf", bbox_inches="tight")

    # Plot the absolute deviation from the centralised cost value - use log scale
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(absolute_deviation[start_iteration:], label=header_names)
    ax.set_yscale('log')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Optimal cost gap, $|\Delta {J}_{\mathrm{op}}|$ [CHF]")
    # ax.set_title("Absolute Deviation from Centralised Cost Value")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(left=start_iteration)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/absolute_deviation_convergence_log.pdf", format="pdf", bbox_inches="tight")


    # Plot the percentual deviation from the centralised cost value - use log scale
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(percent_diff_cost[start_iteration:], label=header_names)
    ax.set_yscale('log')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Optimal cost gap, $\frac{|\Delta {J}_{\mathrm{op}}|}{{J}_{\mathrm{op,optimal}}}$ [$\%$]")
    ax.axhspan(0, 1, color='lightgrey', alpha=0.5, label = r"$< 1\%$")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(left=start_iteration)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/percentual_deviation_convergence_log.pdf", format="pdf", bbox_inches="tight")
    

    # Plot the absolute deviation from the centralised weighted heat value - use log scale
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(abs_diff_w_h_value[start_iteration:], label=header_names)
    ax.set_yscale('log')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Opt. heat value gap, $|\Delta \bar{\lambda}^{\mathrm{H}}|$ [CHF/kWh]")
    # ax.set_title("Absolute Deviation from Centralised Weighted Heat Value")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(left=start_iteration)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/absolute_deviation_w_h_convergence_log.pdf", format="pdf", bbox_inches="tight")

    # Plot the absolute deviation from the centralised weighted electricity value - use log scale
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(abs_diff_w_e_value[start_iteration:], label=header_names)
    ax.set_yscale('log')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Opt. electricity value gap, $|\Delta \bar{\lambda}^{\mathrm{E}}|$ [CHF/kWh]")
    # ax.set_title("Absolute Deviation from Centralised Weighted Electricity Value")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(left=start_iteration)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/absolute_deviation_w_e_convergence_log.pdf", format="pdf", bbox_inches="tight")

    # Plot the absolute deviation from the centralised weighted heat value - use log scale
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(abs_diff_w_h_value[start_iteration:]*100, label=header_names)
    ax.set_yscale('log')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Opt. heat value gap, $|\Delta \bar{\lambda}^{\mathrm{H}}|$ [Rp./kWh]")
    # ax.set_title("Absolute Deviation from Centralised Weighted Heat Value")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(left=start_iteration)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/absolute_deviation_w_h_convergence_log-Rappen.pdf", format="pdf", bbox_inches="tight")

    # Plot the absolute deviation from the centralised weighted electricity value - use log scale
    fig, ax = plt.subplots()
    ax.set_prop_cycle(totalcycle)
    ax.plot(abs_diff_w_e_value[start_iteration:]*100, label=header_names)
    ax.set_yscale('log')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Opt. electricity value gap, $|\Delta \bar{\lambda}^{\mathrm{E}}|$ [Rp./kWh]")
    # ax.set_title("Absolute Deviation from Centralised Weighted Electricity Value")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(left=start_iteration)
    plt.legend(title=r"$\rho^{\mathrm{H}}$, $\rho^{\mathrm{E}}$")
    if save_figures:
        plt.savefig(f"results/{version}/absolute_deviation_w_e_convergence_log-Rappen.pdf", format="pdf", bbox_inches="tight")


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