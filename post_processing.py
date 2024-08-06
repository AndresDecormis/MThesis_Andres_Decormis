import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from parametersv1 import *
import additional_functions as af
import os

#------
# Configuration of this run
af.configure_sns_plots(style='fancy')
name_version = 'centralisedv4-spot_Low_High_Inertia'          # Name of the version of the results

# folders_to_process =   ['results/centralisedv4-bkw_green-high_inertia',
#                         'results/centralisedv4-bkw_green-medium_inertia',
#                         'results/centralisedv4-bkw_green-low_inertia',
#                         'results/centralisedv4-groupe_e_vario_plus-high_inertia',
#                         'results/centralisedv4-groupe_e_vario_plus-medium_inertia',
#                         'results/centralisedv4-groupe_e_vario_plus-low_inertia',  
# ]
# label_folders = ['BKW Green - High inertia',
#                  'BKW Green - Medium inertia',
#                  'BKW Green - Low inertia',
#                  'Vario Plus - High inertia',
#                  'Vario Plus - Medium inertia',
#                  'Vario Plus - Low inertia',
# ]
folders_to_process =   ['results/centralisedv4-spot_spot-high_inertia',
                        'results/centralisedv4-spot_spot-low_inertia',

]
label_folders = ['Spot - High inertia',
                 'Spot - Low inertia']
plot_images = False                          # Plot images in the console
plot_global_images = False                   # Plot global images in the console
save_images = True                         # Save images in the results folder
#------

# Create folder of post-processing in results folder
parent_folder = f"results/post-processing"
folder_version = f"{parent_folder}/{name_version}"
if not os.path.exists(folder_version):
    os.makedirs(folder_version)
else:
    print(f"{folder_version} folder already exists.")

# Create dataframe to store values from the different folders
results_all_df = pd.DataFrame()
main_results_df = pd.DataFrame()

# Save csv files from each folder as a dataframe
for folder in folders_to_process:
    time_data = None
    # Import data of csv file that starts with name 'time_data', but the ending is different for each. There is only one file that starts with 'time_data' in each folder
    for file in os.listdir(folder):
        if file.startswith("time_data"):
            time_data = pd.read_csv(f"{folder}/{file}")
        elif file.startswith("scalar_results"):
            scalar_results = pd.read_csv(f"{folder}/{file}")
        else:
            continue

    subfolder_name = f"{folder_version}/" + folder.split("/")[-1]
    if not os.path.exists(subfolder_name):
        os.makedirs(subfolder_name)
    else:
        print(f"{subfolder_name} folder already exists.")
    
    if time_data.empty:
        raise ValueError(f"Error: time_data is empty for folder {folder}")
    # Plot results for this version
    # Scatter plot of electricity prices and heat values
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="PriceImportElectricity", y="HeatValue", alpha=0.5)
    ax.set_xlabel("Electricity price [CHF/kWh]")
    ax.set_ylabel("Heat value [CHF/kWh]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_heat.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of ambient temperature and heat values
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="TemperatureAmbient", y="HeatValue", alpha=0.5)
    ax.set_xlabel("Ambient temperature [°C]")
    ax.set_ylabel("Heat value [CHF/kWh]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_temperature_heat.pdf", format="pdf", bbox_inches="tight")

    # Scatter and colored plot of ambient temperature and heat values with electricity prices
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="TemperatureAmbient", y="HeatValue", hue="PriceImportElectricity")
    ax.set_xlabel("Ambient temperature [°C]")
    ax.set_ylabel("Heat value [CHF/kWh]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_temperature_heat.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of electricity prices and ambient temperature
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data,  x="TemperatureAmbient", y="PriceImportElectricity",alpha=0.5)
    # Change axes labels
    ax.set_xlabel("Ambient temperature [°C]")
    ax.set_ylabel("Electricity price [CHF/kWh]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_temperature.pdf", format="pdf", bbox_inches="tight")

    # Scatter and colored plot of electricity prices and ambient temperature with heat value
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data,  x="TemperatureAmbient", y="PriceImportElectricity", hue="HeatValue")
    # Change axes labels
    ax.set_xlabel("Ambient temperature [°C]")
    ax.set_ylabel("Electricity price [CHF/kWh]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_temperature_heat.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of heat value and heat demand
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="HeatValue", y="HeatConsumption", alpha=0.5)
    ax.set_xlabel("Heat value [CHF/kWh]")
    ax.set_ylabel("Heat demand [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_heat_heat_demand.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of heat value and average consumer temperature
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="HeatValue", y="TConsAvg", alpha=0.5)
    ax.set_xlabel("Heat value [CHF/kWh]")
    ax.set_ylabel("Average consumer temperature [°C]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_heat_avg_temperature.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of heat demand and ambient temperature
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data,  x="TemperatureAmbient", y="HeatConsumption", alpha=0.5)
    ax.set_xlabel("Ambient temperature [°C]")
    ax.set_ylabel("Heat demand [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_temperature_heat_demand.pdf", format="pdf", bbox_inches="tight")


    # Scatter plot of heat demand and electricity demand
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="HeatConsumption", y="ElectricityDemand", alpha=0.5)
    ax.set_xlabel("Heat demand [kW]")
    ax.set_ylabel("Electricity demand [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_heat_electricity_demand.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of electricity export and electricity export prices
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="PriceExportElectricity", y="ElectricitySupply", alpha=0.5)
    ax.set_xlabel("Electricity export price [CHF/kWh]")
    ax.set_ylabel("Electricity export [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_export.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of electricity import and electricity import prices
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="PriceImportElectricity", y="ElectricityDemand", alpha=0.5)
    ax.set_xlabel("Electricity import price [CHF/kWh]")
    ax.set_ylabel("Electricity import [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_import.pdf", format="pdf", bbox_inches="tight")

    # Histogram of electricity prices and heat values
    fig, ax = plt.subplots()
    sns.histplot(data=time_data, x="PriceImportElectricity", bins=20, label="Electricity price")
    sns.histplot(data=time_data, x="HeatValue", bins=20, label="Heat value")
    ax.set_xlabel("Price or value [CHF/kWh]")
    ax.legend()
    if save_images:
        plt.savefig(f"{subfolder_name}/histogram_electricity_heat.pdf", format="pdf", bbox_inches="tight")

    if plot_images:
        plt.show()

    
    time_data["Version"] = folder.split("/")[-1]
    # Store the results of the version in all results dataframe
    results_all_df = pd.concat([results_all_df, time_data], ignore_index=True)
    
    # Store main results for the version
    main_results_df["Metric"] = scalar_results["Metric"]
    main_results_df[f"Value_{folder}"] = scalar_results["Value"]
    main_results_df["Unit"] = scalar_results["Unit"]

# Plotting scalar results of the different versions
# Converting the data frame for useful version
main_results_df = main_results_df.set_index("Metric")
main_results_df = main_results_df.drop(columns=["Unit"])
main_results_df = main_results_df.transpose()
main_results_df.index.name = "Version"
main_results_df.reset_index(inplace=True)
main_results_df.set_index("Version", inplace=True)


# Box plot of heat values for each folder
fig, ax = plt.subplots()
sns.boxplot(x = "Version", y = "HeatValue", data=results_all_df, ax=ax, palette="crest")
ax.set_xticklabels(label_folders, rotation=60, ha='right')
ax.set_ylabel("Heat value [CHF/kWh]")
if save_images:
    plt.savefig(f"{folder_version}/boxplot_heat_value.pdf", format="pdf", bbox_inches="tight")

# Plot the results of the different folders for comparison
# Bar plots of heat production for the different versions
fig, ax = plt.subplots()
main_results_df[["total_heat_hp_used", "total_heat_fc_used", "total_heat_el_used"]].plot(kind="bar", stacked=True, ax=ax, color=["blue", "red", "green"])
ax.set_ylabel("Heat [kWh]")
ax.set_xticklabels(label_folders, rotation=60, ha='right')
# Adjust the position of the legend
ax.legend(["Heat pump", "Fuel cell", "Electrolyser"])
if save_images:
    plt.savefig(f"{folder_version}/heat_production.pdf", format="pdf", bbox_inches="tight")

# Bar plots of electricity demand and supply for the different versions
# Plots of electricity demand and supply for each folder 
# First making a negative value for the electricity supply
main_results_df["total_electricity_supply_neg"] = -main_results_df["total_electricity_supply"]
fig, ax = plt.subplots()
main_results_df[["total_electricity_hp_demand","total_electricity_el_demand","total_electricity_co_demand","total_electricity_supply_neg"]].plot(kind="bar", stacked=True, ax=ax, color=["blue", "green","gray","red"])
ax.set_ylabel("Electricity [kWh]")
ax.set_xticklabels(label_folders, rotation=60, ha='right')
# Adjust the position of the legend
ax.legend(["Heat pump","Electrolyser","Compressor","Fuel Cell (Generation)"])
if save_images:
    plt.savefig(f"{folder_version}/electricity_demand_supply.pdf", format="pdf", bbox_inches="tight")

# Bar plots of the sources of expenses and revenues for the different versions
# First making revenues negative
main_results_df["electricity_revenue_neg"] = -main_results_df["electricity_revenue"]
main_results_df["hydrogen_revenue_neg"] = -main_results_df["hydrogen_revenue"]

fig, ax = plt.subplots()
main_results_df[["electricity_expenses", "hydrogen_expenses", "electricity_revenue_neg", "hydrogen_revenue_neg"]].plot(kind='bar', stacked=True, ax=ax,color = ["navy", "seagreen", "lightsteelblue", "lightseagreen"])
# Add a bar of the total cost on top of the bars
total_costs = main_results_df["total_cost"]
ax.plot(range(len(label_folders)), total_costs, color='khaki', linestyle="None", marker='o', markersize=6)
# ax.bar(x=range(len(label_folders)), height=total_costs, color='black', alpha=0.5)
ax.set_ylabel('Costs and revenues [CHF]')
ax.set_xticklabels(label_folders, rotation=60, ha='right')
ax.legend(["Total cost","Electricity expenses", "Hydrogen expenses", "Electricity revenue", "Hydrogen revenue"])
if save_images:
    plt.savefig(f"{folder_version}/expenses_revenues.pdf", format="pdf", bbox_inches="tight")

df_to_plot = results_all_df[["HeatValue", "HeatConsumption", "PriceImportElectricity", "TConsAvg", "Version"]]
g = sns.pairplot(df_to_plot, diag_kind='kde',hue="Version")
# Access the legend and modify it
for t, l in zip(g._legend.texts, label_folders):
    t.set_text(l)
# Set the title of the legend
g._legend.set_title("Version")
if save_images:
    plt.savefig(f"{folder_version}/pairplot.png", bbox_inches="tight", dpi=500)


# Histogram of heat values for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="HeatValue", hue="Version", bins=20)
ax.set_xlabel("Heat value [CHF/kWh]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_heat_value.pdf", format="pdf", bbox_inches="tight")

# Histogram of the average consumer temperatures for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="TConsAvg", hue="Version", bins=20)
ax.set_xlabel("Average consumer temperature [°C]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_avg_temperature.pdf", format="pdf", bbox_inches="tight")

# Histogram of the heat demand for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="HeatConsumption", hue="Version", bins=20)
ax.set_xlabel("Heat demand [kW]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_heat_demand.pdf", format="pdf", bbox_inches="tight")

if plot_global_images:
    plt.show()
