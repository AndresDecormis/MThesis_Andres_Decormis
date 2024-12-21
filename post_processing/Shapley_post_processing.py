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


def main():
    """
    Main function to plot shapley comparisons across different scenarios
    """

    af.configure_plots(style='fancy')

    name_version = "PV_capacity_less-11new"
    # Create results folders if they do not exist
    folder_version = f"results/00-ShapleyPostProcessing/{name_version}"
    if not os.path.exists(f"{folder_version}"):
        os.makedirs(f"{folder_version}")
    else:
        print(f"{folder_version} folder already exists.")
    # ---------------------------------------------------------
    
    # Scenarios folders to compare
    # scenarios_folders   = ['results/00-ShapleyValues/shapley_value_v8-PV+BES+TES',
    #                        'results/00-ShapleyValues/shapley_value_v8-PV+BES']
    # label_scenarios     = ['PV+BES+TES', 'PV+BES']

    
    # scenarios_folders   = ["results/00-ShapleyValues/shapley_value_v8-Complete_LowH2Prices",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete_3H2Prices",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete_4H2Prices",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete_5H2Prices",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete_HighH2Prices",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete_7H2Prices",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete_8H2Prices",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete_9H2Prices",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete_CurrentPrices"]
    # label_scenarios     = ['2 CHF/kg', '3 CHF/kg', '4 CHF/kg', '5 CHF/kg', '6 CHF/kg', '7 CHF/kg', '8 CHF/kg', '9 CHF/kg', '10 CHF/kg']
    # color_list          = ['gold', 'darkolivegreen', 'gray', 'blue', 'red']

    

    # scenarios_folders   = ["results/00-ShapleyValues/shapley_value_v8-Complete_CurrentPrices",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete_HighH2Prices",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete_LowH2Prices"]

    # label_scenarios     = [r'$p^{\mathrm{H_2}} = 10 $ CHF/kg',
    #                      r'$p^{\mathrm{H_2}} = 6 $ CHF/kg',
    #                      r'$p^{\mathrm{H_2}} = 2 $ CHF/kg']
    
    # color_list  = ['gold', 'darkolivegreen', 'gray', 'blue', 'red']

    # scenarios_folders   = ["results/00-ShapleyValues/shapley_value_v8-Complete-PV1",
    #                        "results/00-ShapleyValues/shapley_value_v8-Complete-PV10",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete-PV20",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete_CurrentPrices",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete-PV40",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete-PV50",
    #                         "results/00-ShapleyValues/shapley_value_v8-Complete-PV60"]
    # label_scenarios     = ['PV 1m2','PV 10m2' , 'PV 20m2', 'PV 30m2', 'PV 40m2', 'PV 50m2', 'PV 60m2']
    # color_list          = ['gold', 'darkolivegreen', 'gray', 'blue', 'red']

    scenarios_folders   = ["results/00-ShapleyValues/shapley_value_v8-PV11",
                           "results/00-ShapleyValues/shapley_value_v8-PV14",
                           "results/00-ShapleyValues/shapley_value_v8-PV18",
                            "results/00-ShapleyValues/shapley_value_v8-Complete-PV20",
                            "results/00-ShapleyValues/shapley_value_v8-Complete_CurrentPrices",
                            "results/00-ShapleyValues/shapley_value_v8-Complete-PV40",
                            "results/00-ShapleyValues/shapley_value_v8-Complete-PV50",
                            "results/00-ShapleyValues/shapley_value_v8-Complete-PV60"]
    label_scenarios     = [11,14,18,20,30,40,50,60]
    color_list          = ['gold', 'darkolivegreen', 'gray', 'blue', 'red']


    # All possible technologies - naming convention
    technologies = ['PV', 'BES', 'TES', 'EL', 'FC']
    tech_names_plot = ['PV', 'BES', 'TES', 'EL', 'FC']

    plot_stacked_bars       = True
    plot_hydrogen_shapley   = False
    plot_PV_shapley         = True

    # ---------------------------------------------------------
    # Storing the Shapley values of the different scenarios
    # Initialize empty lists to collect data
    tc_data = []
    whv_data = []
    wev_data = []
    # Loop over scenario folders and collect Shapley values
    for i, scenario_folder in enumerate(scenarios_folders):
        # Read Shapley values CSV for the current scenario
        shapley_values_scenario = pd.read_csv(f"{scenario_folder}/shapley_values.csv", index_col=0)
        # Initialize dictionaries for storing this scenario's data
        tc_row  = {'Scenario': label_scenarios[i]}
        whv_row = {'Scenario': label_scenarios[i]}
        wev_row = {'Scenario': label_scenarios[i]}
        # Loop over technologies and get Shapley values for the scenario
        for tech in technologies:
            # Check if tech is used in this scenario; if not, assign 0
            if tech in shapley_values_scenario.index:
                tc_row[tech]  = shapley_values_scenario['Total Cost'].loc[tech]
                whv_row[tech] = shapley_values_scenario['Heat Value'].loc[tech]
                wev_row[tech] = shapley_values_scenario['Electricity Value'].loc[tech]
            else:
                tc_row[tech]  = 0
                whv_row[tech] = 0
                wev_row[tech] = 0
        # Append the rows to the respective lists
        tc_data.append(tc_row)
        whv_data.append(whv_row)
        wev_data.append(wev_row)
    # Convert the collected data into dataframes
    tc_shapley_values_df  = pd.DataFrame(tc_data)
    whv_shapley_values_df = pd.DataFrame(whv_data)
    wev_shapley_values_df = pd.DataFrame(wev_data)

    # Save the dataframes as csv files
    tc_shapley_values_df.to_csv(f"{folder_version}/tc_shapley_values.csv")
    # ---------------------------------------------------------
    if plot_stacked_bars:
        # Plot the Shapley values of the different scenarios in a stacked bar plot
        stack_bar_plot(tc_shapley_values_df, folder_version, "Total cost Shapley Value", tech_names_plot, color_list, label_shapley = r"FOCS value, $\phi^{\mathrm{FOCS}}$ [CHF/year]")
        stack_bar_plot(whv_shapley_values_df, folder_version, "Heat Value Shapley Values", tech_names_plot, color_list, label_shapley = r"FOCS value heat, $\phi^{\mathrm{FOCS}}$ [CHF/year]")
        stack_bar_plot(wev_shapley_values_df, folder_version, "Electricity Value Shapley Values", tech_names_plot, color_list, label_shapley = r"FOCS value electricity, $\phi^{\mathrm{FOCS}}$ [CHF/year]")

    # ---------------------------------------------------------
    if plot_hydrogen_shapley:
        # Hydrogen system Shapley values
        price_list_h2           = [2,3,4,5,6,7,8,9,10]
        el_export_price         = af.get_spot_prices()
        el_import_price_name    = "iwb"
        el_tariff_name          = "power small"
        # Plot the Shapley values of the hydrogen system for the different scenarios
        hydrogen_shapley_plot(el_export_price, el_import_price_name, el_tariff_name, price_list_h2, tc_shapley_values_df, folder_version,color_list,tech_names_plot)
        # simple_heat_shapley_h2(price_list_h2, whv_shapley_values_df, folder_version)

    if plot_PV_shapley:
        PV_capacity_shapleys(tc_shapley_values_df, folder_version, "PV_capacity_Shapley_Values", tech_names_plot,color_list, label_shapley = r"PV FOCS value, $\phi^{\mathrm{FOCS}}_{\mathrm{PV}}$ [CHF/year]")

def stack_bar_plot(shapley_values_df, folder_version, name_shapley_values, tech_names_plot,color_list, label_shapley):
    # Plot the Shapley values of the different scenarios in a stacked bar plot
    # Set the index to 'Scenario' for plotting
    shapley_values_df.set_index('Scenario', inplace=True)
    # Create the plot
    fig, ax = plt.subplots()
    # Plot the dataframe as a stacked bar chart
    shapley_values_df.plot(kind="bar", stacked=True, ax=ax, color = color_list)
    # Set labels and title
    ax.set_ylabel(label_shapley)
    ax.set_xlabel("Scenario")
    ax.set_xticklabels(shapley_values_df.index, rotation=45, ha='right')
    # Adjust the legend and layout
    ax.legend(tech_names_plot)#, bbox_to_anchor=(1.05, 1), loc='upper left')
    # Remove x-axis label
    ax.set_xlabel(None)
    plt.tight_layout()
    plt.savefig(f"{folder_version}/{name_shapley_values}.pdf", format="pdf", bbox_inches="tight")


    
    return None

    
def hydrogen_shapley_plot(el_export_price, el_import_price_name, el_tariff_name, price_list_h2, shapley_values_df, folder_version,color_list,tech_names_plot):
    # Plot the Shapley values of the hydrogen system for the different scenarios
    # Initialize empty lists to collect data
    # h2_shapley_values = list(shapley_values_df.H2System)
    if el_import_price_name         == 'groupe_e':
        p_el_import         = af.get_groupe_e_tariff_data(resolution='hourly', tariff=el_tariff_name)       # Electricity import price [CHF/kWh]
    elif el_import_price_name       == 'bkw':
        p_el_import         = af.get_bkw_tariff_data(tariff=el_tariff_name)       # Electricity import price [CHF/kWh]
    elif el_import_price_name       == 'iwb':
        p_el_import         = af.get_iwb_tariff_data(tariff=el_tariff_name)       # Electricity import price [CHF/kWh]
    elif el_import_price_name       == 'spot':
        if el_tariff_name      == 'spot':
            p_el_import     = af.get_spot_prices()                          # Electricity import price [CHF/kWh]
        elif el_tariff_name    == 'plus_tariff':
            p_el_import     = af.get_spot_prices() + spot_plus_tariff_1     # Electricity import price [CHF/kWh]
    else:
        raise ValueError("Error: Invalid type of electricity price")
    
    # Get H2price to el price ratios
    avg_p_el_import = np.mean(p_el_import)
    min_p_el_import = np.min(p_el_import)
    max_p_el_import = np.max(p_el_import)
    
    h2price_to_el_avg_ratio = [price/avg_p_el_import for price in price_list_h2]
    h2price_to_el_min_ratio = [price/min_p_el_import for price in price_list_h2]
    h2price_to_el_max_ratio = [price/max_p_el_import for price in price_list_h2]

    # # Plot the Shapley values of the hydrogen system for the different avg h2 prices 
    # fig, ax = plt.subplots()
    # ax.plot(h2price_to_el_avg_ratio, h2_shapley_values, label='Average electricity import price')
    # ax.plot(h2price_to_el_min_ratio, h2_shapley_values, label='Minimum electricity import price', linestyle='--')
    # ax.plot(h2price_to_el_max_ratio, h2_shapley_values, label='Maximum electricity import price', linestyle='-.')
    # ax.set_xlabel(r'$\mathrm{H_2}$ price to electricity price ratio')
    # ax.set_ylabel(r'$\mathrm{H_2}$ System Shapley value, $\phi_\mathrm{H_2}$')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(f"{folder_version}/H2System_Shapley_Values_priceratio.pdf", format="pdf", bbox_inches="tight")

    # # Plot the Shapley values of the hydrogen system for the different h2 prices 
    # fig, ax = plt.subplots()
    # ax.plot(price_list_h2, h2_shapley_values, color='#002BFF', label='Shapley values', marker='o', linestyle='--')
    # ax.set_xlabel(r'$\mathrm{H_2}$ price [CHF/kg]')
    # ax.set_ylabel(r'$\mathrm{H_2}$ System Shapley value, $\phi_\mathrm{H_2}$')
    # plt.tight_layout()
    # plt.savefig(f"{folder_version}/H2System_Shapley_Values.pdf", format="pdf", bbox_inches="tight")

    # Plot the Shapley values of each technology for the different h2 prices
    fig, ax = plt.subplots()
    # Give a different marker to each technology
    marker_list = ['o', 's', 'D', 'X', 'P']
    for i, tech in enumerate(tech_names_plot):
        ax.plot(price_list_h2, list(shapley_values_df[tech]), color=color_list[i], label=tech, marker=marker_list[i])
    ax.set_xlabel(r'$\mathrm{H_2}$ price [CHF/kg]')
    ax.set_ylabel(r'FOCS value, $\phi^{\mathrm{FOCS}}$ [CHF/year]')
    ax.legend(tech_names_plot)
    plt.tight_layout()
    plt.savefig(f"{folder_version}/Technology_Shapley_Values_h2prices.pdf", format="pdf", bbox_inches="tight")

    # Plot the Shapley values of each technology for the different h2 prices to elec ratio
    fig, ax = plt.subplots()
    # Give a different marker to each technology
    marker_list = ['o', 's', 'D', 'X', 'P']
    for i, tech in enumerate(tech_names_plot):
        ax.plot(h2price_to_el_max_ratio, list(shapley_values_df[tech]), color=color_list[i], label=tech, marker=marker_list[i])
    ax.set_xlabel(r'$p^\mathrm{H_2}/p^\mathrm{E}_{\mathrm{imp}}$ $\left[\frac{\mathrm{CHF/kg}}{\mathrm{CHF/kWh}} \right]$')
    ax.set_ylabel(r'FOCS value, $\phi^{\mathrm{FOCS}}$ [CHF/year]')
    ax.legend(tech_names_plot)
    plt.tight_layout()
    plt.savefig(f"{folder_version}/Technology_Shapley_Values_h2pricetoelecRatio.pdf", format="pdf", bbox_inches="tight")

    # Plot the Shapley values of each technology for the different h2 prices to elec ratio
    fig, ax = plt.subplots()
    # Give a different marker to each technology
    marker_list = ['o', 's', 'D', 'X', 'P']
    for i, tech in enumerate(tech_names_plot):
        ax.plot(h2price_to_el_min_ratio, list(shapley_values_df[tech]), color=color_list[i], label=tech, marker=marker_list[i])
    ax.set_xlabel(r'$p^\mathrm{H_2}/p^\mathrm{E,low}_{\mathrm{imp}}$ $\left[\frac{\mathrm{CHF/kg}}{\mathrm{CHF/kWh}} \right]$')
    ax.set_ylabel(r'FOCS value, $\phi^{\mathrm{FOCS}}$ [CHF/year]')
    ax.legend(tech_names_plot)
    plt.tight_layout()
    plt.savefig(f"{folder_version}/Technology_Shapley_Values_h2pricetoelecRatio_Low.pdf", format="pdf", bbox_inches="tight")
    return None

def PV_capacity_shapleys(shapley_values_df, folder_version, name_shapley_values, tech_names_plot,color_list, label_shapley):
    PV_shapleys = list(shapley_values_df.PV)
    PV_capacity_list = [11,14,18,20,30,40,50,60]
    # Plot the Shapley values of the PV for the different scenarios
    fig, ax = plt.subplots()
    ax.plot(PV_capacity_list, PV_shapleys, color='darkgoldenrod', marker='o', linestyle='-')
    ax.set_xlabel(r'$\mathrm{PV}$ capacity [m$^2$]')
    ax.set_ylabel(label_shapley)
    plt.tight_layout()
    plt.savefig(f"{folder_version}/{name_shapley_values}.pdf", format="pdf", bbox_inches="tight")


    # Plot the normalised Shapley values of the PV for the different scenarios
    PV_shapleys_normalised_by_area = [shapley/PV_capacity for shapley, PV_capacity in zip(PV_shapleys, PV_capacity_list)]
    fig, ax = plt.subplots()
    ax.plot(PV_capacity_list, PV_shapleys_normalised_by_area, color='darkgoldenrod', marker='o', linestyle='-')
    ax.set_xlabel(r'$\mathrm{PV}$ capacity [m$^2$]')
    ax.set_ylabel(r"Specific PV FOCS value, $\frac{\phi^{\mathrm{FOCS}}_{\mathrm{PV}}}{A_{\mathrm{PV}}}$ $\left[\frac{\mathrm{CHF/year}}{\mathrm{m}^2}\right]$")
    plt.tight_layout()
    plt.savefig(f"{folder_version}/{name_shapley_values}-normalised.pdf", format="pdf", bbox_inches="tight")

    BES_shapleys = list(shapley_values_df.BES)
    BES_capacity_list = 30 * np.ones(len(PV_capacity_list))
    BES_shapleys_normalised_by_area = [shapley/BES_capacity for shapley, BES_capacity in zip(BES_shapleys, BES_capacity_list)]
    # Plot the normalised Shapley values of the PV and of the BES in a separate axis for the different scenarios
    fig, ax = plt.subplots()
    ax.plot(PV_capacity_list, PV_shapleys_normalised_by_area, color='darkgoldenrod', marker='o', linestyle='-', label='PV')
    ax.set_xlabel(r'$\mathrm{PV}$ capacity [m$^2$]')
    ax.set_ylabel(r"Specific PV FOCS value, $\frac{\phi^{\mathrm{FOCS}}_{\mathrm{PV}}}{A_{\mathrm{PV}}}$ $\left[\frac{\mathrm{CHF/year}}{\mathrm{m}^2}\right]$")
    ax2 = ax.twinx()
    ax2.plot(PV_capacity_list, BES_shapleys_normalised_by_area, color='darkolivegreen', marker='x', linestyle='--', label='BES')
    ax2.set_ylabel(r"Specific BES FOCS value $\frac{\phi^{\mathrm{FOCS}}_{\mathrm{BES}}}{E^{\mathrm{max}}_{\mathrm{BES}}}$ $\left[\frac{\mathrm{CHF/year}}{\mathrm{kWh}}\right]$")
    plt.tight_layout()
    # Add legends for both axes
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig(f"{folder_version}/{name_shapley_values}-normalised-PV-BES.pdf", format="pdf", bbox_inches="tight")



    return None

def simple_heat_shapley_h2(price_list_h2, shapley_values_df, folder_version):
    h2_shapley_values = list(shapley_values_df.H2System)
    # Plot the Shapley values of the hydrogen system for the different h2 prices 
    fig, ax = plt.subplots()
    ax.plot(price_list_h2, h2_shapley_values, color='#BB0000', label='Shapley values', marker='o', linestyle='--')
    ax.set_xlabel(r'$\mathrm{H_2}$ price [CHF/kg]')
    ax.set_ylabel(r'$\mathrm{H_2}$ System Heat Shapley value, $\phi^\mathrm{H}_\mathrm{H_2}$')
    plt.tight_layout()
    plt.savefig(f"{folder_version}/H2System_WHV_Shapley_Values.pdf", format="pdf", bbox_inches="tight")


    return None

if __name__ == "__main__":
    main()  # Run the main function

    

