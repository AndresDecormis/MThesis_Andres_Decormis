import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parametersv1 import *
import os
import additional_functions as af


#------
# Configuration of this run
af.configure_plots(style='fancy')
name_version    = 'electricity_prices'          # Name of the version of the results
save_images     = True                         # Save images in the results folder
plot_images     = True                          # Plot images in the console
#------

# Create results folders if they do not exist
folder_version = f"results/{name_version}"
if not os.path.exists(f"{folder_version}"):
    os.makedirs(f"{folder_version}")
else:
    print(f"{folder_version} folder already exists.")

# Import data
p_el_spot       = af.get_spot_prices()                          # Electricity export price [CHF/kWh]
p_el_vario_grid = af.get_groupe_e_tariff_data(resolution='hourly', tariff='vario_grid')       # Electricity import price [CHF/kWh]
p_el_vario_plus = af.get_groupe_e_tariff_data(resolution='hourly', tariff='vario_plus')       # Electricity import price [CHF/kWh]
p_el_dt_plus    = af.get_groupe_e_tariff_data(resolution='hourly', tariff='dt_plus')       # Electricity import price [CHF/kWh]
p_el_bkw_green  = af.get_bkw_tariff_data(tariff='green')       # Electricity import price [CHF/kWh]
p_el_bkw_blue   = af.get_bkw_tariff_data(tariff='blue')       # Electricity import price [CHF/kWh]
p_el_bkw_grey   = af.get_bkw_tariff_data(tariff='grey')       # Electricity import price [CHF/kWh]

# Extract useful statistics from each price
# Mean electricity prices
p_el_spot_mean          = np.mean(p_el_spot)
p_el_vario_grid_mean    = np.mean(p_el_vario_grid)
p_el_vario_plus_mean    = np.mean(p_el_vario_plus)
p_el_dt_plus_mean       = np.mean(p_el_dt_plus)
p_el_bkw_green_mean     = np.mean(p_el_bkw_green)
p_el_bkw_blue_mean      = np.mean(p_el_bkw_blue)
p_el_bkw_grey_mean      = np.mean(p_el_bkw_grey)
# Standard deviation of electricity prices
p_el_spot_std           = np.std(p_el_spot)
p_el_vario_grid_std     = np.std(p_el_vario_grid)
p_el_vario_plus_std     = np.std(p_el_vario_plus)
p_el_dt_plus_std        = np.std(p_el_dt_plus)
p_el_bkw_green_std      = np.std(p_el_bkw_green)
p_el_bkw_blue_std       = np.std(p_el_bkw_blue)
p_el_bkw_grey_std       = np.std(p_el_bkw_grey)

# Save the statistics in a csv file
df = pd.DataFrame()
df['Tariff name'] = ['Spot', 'Vario Grid', 'Vario Plus', 'DT Plus', 'BKW Green', 'BKW Blue', 'BKW Grey']
df['Mean electricity price [CHF/kWh]'] = [p_el_spot_mean, p_el_vario_grid_mean, p_el_vario_plus_mean, p_el_dt_plus_mean, p_el_bkw_green_mean, p_el_bkw_blue_mean, p_el_bkw_grey_mean]
df['Standard deviation [CHF/kWh]'] = [p_el_spot_std, p_el_vario_grid_std, p_el_vario_plus_std, p_el_dt_plus_std, p_el_bkw_green_std, p_el_bkw_blue_std, p_el_bkw_grey_std]
df.to_csv(f"{folder_version}/mean_electricity_prices.csv", index=False)

# Make hourly time index for the plots
time_index = pd.date_range(start='2023-01-01 00:00:00', end='2023-12-31 23:00:00', freq='H')

# Plot electricity prices
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(time_index, p_el_spot, label='Spot', color='black', linestyle='dashed')
ax.plot(time_index, p_el_vario_grid, label='Vario Grid', color='orange', linestyle='dotted')
ax.plot(time_index, p_el_vario_plus, label='Vario Plus', color='purple', linestyle='dashdot')
ax.plot(time_index, p_el_dt_plus, label='DT Plus', color='red')
ax.plot(time_index, p_el_bkw_green, label='BKW Green', color='green')
ax.plot(time_index, p_el_bkw_blue, label='BKW Blue', color='blue')
ax.plot(time_index, p_el_bkw_grey, label='BKW Grey', color='grey')
ax.set_xlabel('Time')
ax.set_ylabel('Electricity price [CHF/kWh]')
ax.legend()
if save_images:
    plt.savefig(f"{folder_version}/electricity_prices_year.pdf", format = 'pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_index, p_el_spot, label='Spot', color='black', linestyle='dashed')
ax.plot(time_index, p_el_vario_grid, label='Vario Grid', color='orange', linestyle='dotted')
ax.plot(time_index, p_el_vario_plus, label='Vario Plus', color='purple', linestyle='dashdot')
ax.plot(time_index, p_el_dt_plus, label='DT Plus', color='red')
ax.plot(time_index, p_el_bkw_green, label='BKW Green', color='green')
ax.plot(time_index, p_el_bkw_blue, label='BKW Blue', color='blue')
ax.plot(time_index, p_el_bkw_grey, label='BKW Grey', color='grey')
ax.set_xlabel('Time')
ax.set_ylabel('Electricity price [CHF/kWh]')
ax.legend()
ax.set_xlim([time_index[0], time_index[23]])
ax.set_xticks(time_index[0:24])
plt.xticks(rotation=45)
ax.set_xticklabels(time_index[0:24].strftime('%H:%M'))
if save_images:
    plt.savefig(f"{folder_version}/electricity_prices_first_day.pdf", format = 'pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_index, p_el_spot, label='Spot', color='black', linestyle='dashed')
ax.plot(time_index, p_el_vario_grid, label='Vario Grid', color='orange', linestyle='dotted')
ax.plot(time_index, p_el_vario_plus, label='Vario Plus', color='purple', linestyle='dashdot')
ax.plot(time_index, p_el_dt_plus, label='DT Plus', color='red')
ax.plot(time_index, p_el_bkw_green, label='BKW Green', color='green')
ax.plot(time_index, p_el_bkw_blue, label='BKW Blue', color='blue')
ax.plot(time_index, p_el_bkw_grey, label='BKW Grey', color='grey')
ax.set_xlabel('Time')
ax.set_ylabel('Electricity price [CHF/kWh]')
ax.legend()
ax.set_xlim([time_index[8520], time_index[8543]])
ax.set_xticks(time_index[8520:8544])
plt.xticks(rotation=45)
ax.set_xticklabels(time_index[8520:8544].strftime('%H:%M'))
if save_images:
    plt.savefig(f"{folder_version}/electricity_prices_winter_day.pdf", format = 'pdf', bbox_inches='tight')


# Plot another random day
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_index, p_el_spot, label='Spot', color='black', linestyle='dashed')
ax.plot(time_index, p_el_vario_grid, label='Vario Grid', color='orange', linestyle='dotted')
ax.plot(time_index, p_el_vario_plus, label='Vario Plus', color='purple', linestyle='dashdot')
ax.plot(time_index, p_el_dt_plus, label='DT Plus', color='red')
ax.plot(time_index, p_el_bkw_green, label='BKW Green', color='green')
ax.plot(time_index, p_el_bkw_blue, label='BKW Blue', color='blue')
ax.plot(time_index, p_el_bkw_grey, label='BKW Grey', color='grey')
ax.set_xlabel('Time')
ax.set_ylabel('Electricity price [CHF/kWh]')
ax.legend()
ax.set_xlim([time_index[4128], time_index[4151]])
ax.set_xticks(time_index[4128:4152])
plt.xticks(rotation=45)
ax.set_xticklabels(time_index[4128:4152].strftime('%H:%M'))
if save_images:
    plt.savefig(f"{folder_version}/electricity_prices_summer_day.pdf", format = 'pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_index, p_el_spot, label='Spot', color='black', linestyle='dashed')
ax.plot(time_index, p_el_vario_grid, label='Vario Grid', color='orange', linestyle='dotted')
ax.plot(time_index, p_el_vario_plus, label='Vario Plus', color='purple', linestyle='dashdot')
ax.plot(time_index, p_el_dt_plus, label='DT Plus', color='red')
ax.plot(time_index, p_el_bkw_green, label='BKW Green', color='green')
ax.plot(time_index, p_el_bkw_blue, label='BKW Blue', color='blue')
ax.plot(time_index, p_el_bkw_grey, label='BKW Grey', color='grey')
ax.set_xlabel('Time')
ax.set_ylabel('Electricity price [CHF/kWh]')
ax.legend()
ax.set_xlim([time_index[0], time_index[167]])
plt.xticks(rotation=45)
if save_images:
    plt.savefig(f"{folder_version}/electricity_prices_winter_week.pdf", format = 'pdf', bbox_inches='tight')

# Plot summer week
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_index, p_el_spot, label='Spot', color='black', linestyle='dashed')
ax.plot(time_index, p_el_vario_grid, label='Vario Grid', color='orange', linestyle='dotted')
ax.plot(time_index, p_el_vario_plus, label='Vario Plus', color='purple', linestyle='dashdot')
ax.plot(time_index, p_el_dt_plus, label='DT Plus', color='red')
ax.plot(time_index, p_el_bkw_green, label='BKW Green', color='green')
ax.plot(time_index, p_el_bkw_blue, label='BKW Blue', color='blue')
ax.plot(time_index, p_el_bkw_grey, label='BKW Grey', color='grey')
ax.set_xlabel('Time')
ax.set_ylabel('Electricity price [CHF/kWh]')
ax.legend()
ax.set_xlim([time_index[4128], time_index[4295]])
plt.xticks(rotation=45)
if save_images:
    plt.savefig(f"{folder_version}/electricity_prices_summer_week.pdf", format = 'pdf', bbox_inches='tight')

# Plot histograms of electricity prices
n_bins = 30
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(p_el_spot, bins=n_bins, alpha=0.2, label='Spot', color='black')
ax.hist(p_el_vario_grid, bins=n_bins, alpha=0.5, label='Vario Grid', color='orange')
ax.hist(p_el_vario_plus, bins=n_bins, alpha=0.5, label='Vario Plus', color='purple')
# plot vertical lines at determined values
ax.axvline(p_el_dt_plus[0], color='red', alpha = 0.8, label='DT Plus', linewidth=3)
ax.axvline(p_el_dt_plus[10], color='red', alpha = 0.8, linewidth=3)
ax.axvline(p_el_bkw_green[0], color='green', alpha = 1, label='BKW Green', linewidth=3)
ax.axvline(p_el_bkw_green[10], color='green', alpha = 1, linewidth=3)
ax.axvline(p_el_bkw_blue[0], color='blue', alpha = 1, label='BKW Blue', linewidth=3)
ax.axvline(p_el_bkw_blue[10], color='blue', alpha = 1, linewidth=3)
ax.axvline(p_el_bkw_grey[0], color='black', alpha = 1, label='BKW Grey', linewidth=3)
ax.axvline(p_el_bkw_grey[10], color='black', alpha = 0.8, linewidth=3)
ax.set_xlabel('Electricity price [CHF/kWh]')
ax.set_ylabel('Frequency')
ax.legend()
if save_images:
    plt.savefig(f"{folder_version}/electricity_prices_histogram.pdf", format = 'pdf', bbox_inches='tight')

if plot_images:
    plt.show()