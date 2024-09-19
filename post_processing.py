import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Archive.parametersv1 import *
import additional_functions as af
import os

# ------------------------------
# Configuration of this run
# ------------------------------
af.configure_sns_plots(style='fancy')
# ------------------------------------------------------------------------------------------
name_version = 'distributed-test-comparison-07sep'                # TODO Name of the version of the results
# ------------------------------------------------------------------------------------------
# Folders to process
folders_to_process =   ['results/distributedv8-iwb_power_small-Medium_TI-True_EC-True_PV-True_BES-True_TES-True_EL-True_FC-Medium_TF-True_H2-current_H2Price-3_PWA_07sep/rho_0_05U0_05',
                        'results/centralisedv8-iwb_power_small-Medium_TI-Medium_TF-True_EC-True_PV-True_BES-True_TES-True_EL-True_FC-True_H2-current_H2Price-3_PWA_07sep']     # TODO: Folders to process      
label_folders = ['distributed',
                 'centralised']                         # TODO: Label of the folders (in order)
plot_images             = False                             # Plot images in the console
plot_global_images      = False                             # Plot global images in the console
save_images             = True                              # Save images in the results folder
# ------------------------------

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
    
    # ------- Plot results for this version -------
    #----- Heat values in axis of scatter plots
    # Scatter plot of electricity prices and heat values
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="PriceImportElectricity", y="HeatValue", alpha=0.5)
    ax.set_xlabel("Electricity price [CHF/kWh]")
    ax.set_ylabel("Heat value [CHF/kWh]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_heatvalue.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of ambient temperature and heat values
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="TemperatureAmbient", y="HeatValue", alpha=0.5)
    ax.set_xlabel("Ambient temperature [°C]")
    ax.set_ylabel("Heat value [CHF/kWh]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_temperature_heatvalue.pdf", format="pdf", bbox_inches="tight")

    # Scatter and colored plot of ambient temperature and heat values with electricity prices
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="TemperatureAmbient", y="HeatValue", hue="PriceImportElectricity")
    ax.set_xlabel("Ambient temperature [°C]")
    ax.set_ylabel("Heat value [CHF/kWh]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_temperature_heatvalue_hue_pelimport.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of heat value and heat demand
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="HeatValue", y="HeatDemand", alpha=0.5)
    ax.set_xlabel("Heat value [CHF/kWh]")
    ax.set_ylabel("Heat demand [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_heatvalue_heatdemand.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of heat value and average consumer temperature
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="HeatValue", y="TConsAvg", alpha=0.5)
    ax.set_xlabel("Heat value [CHF/kWh]")
    ax.set_ylabel("Average consumer temperature [°C]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_heat_avg_temperature.pdf", format="pdf", bbox_inches="tight")

    #----- Price import electricity in axis of scatter plots
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

    
    #----- Price export electricity in axis of scatter plots
    # Scatter plot of price export electricity and heat values
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="PriceExportElectricity", y="HeatValue", alpha=0.5)
    ax.set_xlabel("Electricity export price [CHF/kWh]")
    ax.set_ylabel("Heat value [CHF/kWh]")   
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_pel_export_heatvalue.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of price export electricity and heat values with hue heat demand
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="PriceExportElectricity", y="HeatValue", hue="HeatDemand", palette="rocket_r")
    ax.set_xlabel("Electricity export price [CHF/kWh]")
    ax.set_ylabel("Heat value [CHF/kWh]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_pel_export_heatvalue_hue_heatdemand.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of heat demand and ambient temperature
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data,  x="TemperatureAmbient", y="HeatDemand", alpha=0.5)
    ax.set_xlabel("Ambient temperature [°C]")
    ax.set_ylabel("Heat demand [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_temperature_heat_demand.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of electricity exported and electricity export prices
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="PriceExportElectricity", y="ElectricityExported", alpha=0.5)
    ax.set_xlabel("Electricity export price [CHF/kWh]")
    ax.set_ylabel("Electricity exported [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_exported.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of heat demand and electricity demand
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="HeatDemand", y="ElectricityForHeatDemand", alpha=0.5)
    ax.set_xlabel("Heat demand [kW]")
    ax.set_ylabel("Electricity demand for heat production [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_heat_electricity_demand.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of electricity generated and electricity export prices
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="PriceExportElectricity", y="ElectricityGenerated", alpha=0.5)
    ax.set_xlabel("Electricity export price [CHF/kWh]")
    ax.set_ylabel("Electricity generated [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_generated.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of electricity generated by fuel cell and pv separately and electricity export prices
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="PriceExportElectricity", y="ElectricityFCProduced", alpha=0.5, label="Fuel cell", color="orangered")
    sns.scatterplot(data=time_data, x="PriceExportElectricity", y="ElectricityPVProduced", alpha=0.5, label="PV", color="gold")
    ax.set_xlabel("Electricity export price [CHF/kWh]")
    ax.set_ylabel("Electricity generated [kW]")
    ax.legend()
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_generated_fc_pv.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of electricity import and electricity import prices
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="PriceImportElectricity", y="ElectricityImported", alpha=0.5)
    ax.set_xlabel("Electricity import price [CHF/kWh]")
    ax.set_ylabel("Electricity import [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_imported.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of electricity import prices, heat consumption with the heat generated from hydrogen as hue
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="PriceImportElectricity", y="HeatConsumption", hue="HeatH2", palette="crest")
    ax.set_xlabel("Electricity import price [CHF/kWh]")
    ax.set_ylabel("Heat demand [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_heat_h2.pdf", format="pdf", bbox_inches="tight")

    # Histogram of electricity prices and heat values
    fig, ax = plt.subplots()
    sns.histplot(data=time_data, x="PriceImportElectricity", bins=20, label="Electricity price", stat="probability")
    sns.histplot(data=time_data, x="HeatValue", bins=20, label="Heat value", stat="probability")
    ax.set_xlabel("Price or value [CHF/kWh]")
    ax.legend()
    if save_images:
        plt.savefig(f"{subfolder_name}/histogram_electricity_heat.pdf", format="pdf", bbox_inches="tight")

    # ------ Heat generation -------
    # Scatter plot of electricity consumption and heat generation for HP and EL
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="LoadHeatPump", y="HeatHP", alpha=0.5, label="Heat pump", color="darkkhaki")
    sns.scatterplot(data=time_data, x="LoadElectrolyser", y="HeatEl", alpha=0.5, label="Electrolyser", color="royalblue")
    ax.set_xlabel("Electricity demand [kW]")
    ax.set_ylabel("Heat generation [kW]")
    ax.legend()
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_heat_hp_el.pdf", format="pdf", bbox_inches="tight")



    # ------ Operations of the system ------
    # ------ Hydrogen
    # Scatter plot of hydrogen production and electrolyser load
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="LoadElectrolyser", y="H2Produced", alpha=0.5)
    ax.set_xlabel("Electrolyser load [kW]")
    ax.set_ylabel("Hydrogen produced [kg/h]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electrolyser_hydrogen.pdf", format="pdf", bbox_inches="tight")
    
    # Scatter plot of hydorgen consumption and fuel cell production
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="H2Consumed", y="ElectricityFCProduced", alpha=0.5, color="orangered")
    ax.set_xlabel("Hydrogen consumed [kg/h]")
    ax.set_ylabel("Electricity produced [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_hydrogen_consumed_fc_production.pdf", format="pdf", bbox_inches="tight")

        # Scatter plot of heat generation and electricity generation for FC
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="ElectricityFCProduced", y="HeatFC", alpha=0.5, color="orangered")
    ax.set_xlabel("Electricity produced [kW]") 
    ax.set_ylabel("Heat generation [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electricity_heat_fc.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of heat generated and hydrogen used in fuel cell
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="H2FCUsed", y="HeatFC", alpha=0.5, color="orangered")
    ax.set_xlabel("Hydrogen used [kg/h]")
    ax.set_ylabel("Heat generation [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_hydrogen_heat_fc.pdf", format="pdf", bbox_inches="tight")

    # Scatter plot of heat generated and load of electrolyser
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="LoadElectrolyser", y="HeatEl", alpha=0.5, color="royalblue")
    ax.set_xlabel("Electrolyser load [kW]")
    ax.set_ylabel("Heat generation [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electrolyser_heat_el.pdf", format="pdf", bbox_inches="tight")


    # Scatter plot of total heat generated and load of electrolyser
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="LoadElectrolyser", y="HeatElGen", alpha=0.5, color="royalblue")
    ax.set_xlabel("Electrolyser load [kW]")
    ax.set_ylabel("Total heat generation [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_electrolyser_heat_gen_el.pdf", format="pdf", bbox_inches="tight")# Scatter plot of total heat generated and electricity used by electrolyser
    
    # Scatter plot of total heat generated and hydrogen used by fuel cell
    fig, ax = plt.subplots()
    sns.scatterplot(data=time_data, x="H2FCUsed", y="HeatFCGen", alpha=0.5, color="orangered")
    ax.set_xlabel("Hydrogen used [kg/h]")
    ax.set_ylabel("Total heat generation [kW]")
    if save_images:
        plt.savefig(f"{subfolder_name}/scatter_hydrogen_heat_gen_fc.pdf", format="pdf", bbox_inches="tight")

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
main_results_df[["total_heat_hp_used", "total_heat_fc_used", "total_heat_el_used"]].plot(kind="bar", stacked=True, ax=ax, color=["darkkhaki", "orangered", "royalblue"])
ax.set_ylabel("Heat [kWh]")
ax.set_xticklabels(label_folders, rotation=60, ha='right')
# Adjust the position of the legend
ax.legend(["Heat pump", "Fuel cell", "Electrolyser"])
if save_images:
    plt.savefig(f"{folder_version}/heat_production.pdf", format="pdf", bbox_inches="tight")

# Bar plots of electricity demand and supply for the different versions
# Plots of electricity demand and supply for each folder 
# First making a negative value for the electricity supply
main_results_df["total_electricity_fc_produced_neg"] = -main_results_df["total_electricity_fc_produced"]
main_results_df["total_electricity_pv_produced_neg"] = -main_results_df["total_electricity_pv_produced"]
fig, ax = plt.subplots()
main_results_df[["total_electricity_hp_demand","total_electricity_el_demand","total_electricity_co_demand","total_electricity_fc_produced_neg", "total_electricity_pv_produced_neg"]].plot(kind="bar", stacked=True, ax=ax, color=["darkkhaki", "royalblue","gray","orangered","gold"])
ax.set_ylabel("Electricity [kWh]")
ax.set_xticklabels(label_folders, rotation=60, ha='right')
# Adjust the legend
ax.legend(["Heat pump","Electrolyser","Compressor","Fuel Cell (Generation)", "PV (Generation)"])
if save_images:
    plt.savefig(f"{folder_version}/electricity_demand_generation.pdf", format="pdf", bbox_inches="tight")


# Bar plots of electricity import and export for the different versions
# First making a negative value for the electricity export
main_results_df["total_electricity_export_neg"] = -main_results_df["total_electricity_exported"]
fig, ax = plt.subplots()
main_results_df[["total_electricity_imported","total_electricity_export_neg"]].plot(kind="bar", stacked=True, ax=ax, color=["steelblue", "sienna"])
ax.set_ylabel("Electricity [kWh]")
ax.set_xticklabels(label_folders, rotation=60, ha='right')
# Adjust the legend
ax.legend(["Electricity imported","Electricity exported"])
if save_images:
    plt.savefig(f"{folder_version}/electricity_import_export.pdf", format="pdf", bbox_inches="tight")

# Bar plots of electricity import and export for the different versions
# Bar plots next to each other
fig, ax = plt.subplots()
main_results_df[["total_electricity_imported","total_electricity_exported"]].plot(kind="bar", ax=ax, color=["slategrey", "peru"])
ax.set_ylabel("Electricity [kWh]")
ax.set_xticklabels(label_folders, rotation=60, ha='right')
# Adjust the legend
ax.legend(["Electricity imported","Electricity exported"])
if save_images:
    plt.savefig(f"{folder_version}/electricity_import_export_next.pdf", format="pdf", bbox_inches="tight")

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

# # A pairplot of the results
# df_to_plot = results_all_df[["HeatValue", "HeatConsumption", "PriceImportElectricity", "TConsAvg", "Version"]]
# g = sns.pairplot(df_to_plot, diag_kind='kde',hue="Version")
# # Access the legend and modify it
# for t, l in zip(g._legend.texts, label_folders):
#     t.set_text(l)
# # Set the title of the legend
# g._legend.set_title("Version")
# g._legend.get_frame().set_edgecolor('black')  # Set edge color to black
# g._legend.get_frame().set_linewidth(1.5)  # Set the line width of the frame
# if save_images:
#     plt.savefig(f"{folder_version}/pairplot.png", bbox_inches="tight", dpi=500)


# Histogram of heat values for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="HeatValue", hue="Version", bins=20)
ax.set_xlabel("Heat value [CHF/kWh]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_heat_value.pdf", format="pdf", bbox_inches="tight")

# Histogram of the electricity prices for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="PriceImportElectricity", hue="Version", bins=20, stat="probability")
ax.set_xlabel("Electricity price [CHF/kWh]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_electricity_price.pdf", format="pdf", bbox_inches="tight")

# Histogram of the average consumer temperatures for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="TConsAvg", hue="Version", bins=20, stat="probability")
ax.set_xlabel("Average consumer temperature [°C]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_avg_temperature.pdf", format="pdf", bbox_inches="tight")

# Histogram of the heat demand for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="HeatDemand", hue="Version", bins=20, stat="probability")
ax.set_xlabel("Heat demand [kW]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_heat_demand.pdf", format="pdf", bbox_inches="tight")

# Histogram of the electricity demand for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="ElectricityForHeatDemand", hue="Version", bins=20, stat="probability")
ax.set_xlabel("Electricity demand for heating [kW]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_electricity_f_heat_demand.pdf", format="pdf", bbox_inches="tight")

# Histogram of the electricity supply for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="ElectricityGenerated", hue="Version", bins=20, stat="probability")
ax.set_xlabel("Electricity generated [kW]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_electricity_generated.pdf", format="pdf", bbox_inches="tight")

# Histogram of the electricity import for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="ElectricityImported", hue="Version", bins=20, stat="probability")
ax.set_xlabel("Electricity imported [kW]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_electricity_imported.pdf", format="pdf", bbox_inches="tight")

# Histogram of the electricity export for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="ElectricityExported", hue="Version", bins=20, stat="probability")
ax.set_xlabel("Electricity exported [kW]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_electricity_exported.pdf", format="pdf", bbox_inches="tight")


# Histogram of the rate of change of electricity demand for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="RateElecFHeatDemand", hue="Version", bins=20, stat="probability")
ax.set_xlabel("Rate of change in electricity demand for heating [kW/h]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_rate_electricity_demand_f_heat.pdf", format="pdf", bbox_inches="tight")

# Histogram of the rate of change of heat demand for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="RateHeatDemand", hue="Version", bins=20, stat="probability")
ax.set_xlabel("Rate of change in heat demand [kW/h]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_rate_heat_demand.pdf", format="pdf", bbox_inches="tight")

# Histogram of the rate of change of electricity supply for each version
fig, ax = plt.subplots()
sns.histplot(data=results_all_df, x="RateElecGenerated", hue="Version", bins=20, stat="probability")
ax.set_xlabel("Rate of change in electricity generated [kW/h]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_rate_electricity_generated.pdf", format="pdf", bbox_inches="tight")

# Histogram of the rate of change of electricity demand, only for positive values
fig, ax = plt.subplots()
sns.histplot(data=results_all_df[results_all_df["RateElecFHeatDemand"] > 0], x="RateElecFHeatDemand", hue="Version", bins=30, stat="probability")
ax.set_xlabel("Rate of change in electricity demand for heating [kW/h]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_rate_electricity_demand_positive.pdf", format="pdf", bbox_inches="tight")

# Histogram of the rate of change of heat demand, only for positive values
fig, ax = plt.subplots()
sns.histplot(data=results_all_df[results_all_df["RateHeatDemand"] > 0], x="RateHeatDemand", hue="Version", bins=30, stat="probability")
ax.set_xlabel("Rate of change in heat demand [kW/h]")
ax.legend(title="Version", labels=label_folders)
if save_images:
    plt.savefig(f"{folder_version}/histogram_rate_heat_demand_positive.pdf", format="pdf", bbox_inches="tight")

if plot_global_images:
    plt.show()
