# All the functions that are used in the main code and not in the main code itself are stored here.

import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from cycler import cycler
import numpy as np

def get_hydrogen_price():
    """
    Function to get the hydrogen price - source: https://data.sccer-jasm.ch/import-prices/2020-08-01/
    :return: float with the hydrogen price
    """
    hydrogen_market_price = 1.2        # Hydrogen market price [CHF/kgH2] - 3.3: source: Bernardino
    return hydrogen_market_price       # Hydrogen price [CHF/kg]

def get_spot_prices():
    """
    Function to get the spot prices - source https://transparency.entsoe.eu/
    Remuneration of produced electricity is based on spot prices by SwissIX - https://www.iwb.ch/servicecenter/stromtarife/rueckliefertarif
    :return: numpy array with the spot prices
    """
    data = pd.read_csv("data/data2023.csv")                     # Import data
    data["Electricity_priceCHF"] = data["Electricity_priceCHF"] * 1e-3 # Convert to CHF/kWh
    return data["Electricity_priceCHF"].to_numpy()       # Electricity price [CHF/kWh]

def get_temperature_data(city: str = 'Basel'):
    """
    Function to get the temperature data - Zurich source: https://opendata.swiss/en/dataset/stundlich-aktualisierte-meteodaten-seit-1992
                                          Basel source: https://climate.onebuilding.org/WMO_Region_6_Europe/CHE_Switzerland/index.html#IDBS_Basel-Stadt-
    :return: numpy array with the temperature data
    """
    data = pd.read_csv("data/data2023.csv")                     # Import data
    if city == 'Zurich':
        return data["TemperatureC"].to_numpy()                  # Temperature [°C]
    elif city == 'Basel':
        return data["Temperature_Basel_degC"].to_numpy()        # Temperature [°C]
    else:
        raise ValueError("Error: Invalid city")

def get_bkw_tariff_data(tariff: str = 'green'):
    """
    Function to get the simple tariff data - source: https://www.bkw.ch/fileadmin/user_upload/03_Energie/03_06_Gesetzliche_Publikationen/Tarife___Tarifanpassungen/Tarifblaetter_2024/20989_TB_Stromprodukte_Gross-Industriekunden__2024_DE_fertig.pdf
    :param tariff: str, 'green' or 'blue' or 'grey'
    :return: numpy array with the tariff data
    """
    if tariff == 'green':
        high_tariff = 0.1413 # High tariff [CHF/kWh]
        low_tariff  = 0.1111 # Low tariff [CHF/kWh]
    elif tariff == 'blue':
        high_tariff = 0.1144 # High tariff [CHF/kWh]
        low_tariff  = 0.0842 # Low tariff [CHF/kWh]
    elif tariff == 'grey':
        high_tariff = 0.1036 # High tariff [CHF/kWh]
        low_tariff  = 0.0735 # Low tariff [CHF/kWh]
    else:
        raise ValueError("Error: Invalid type of tariff")
    
    # Make the yearly tariff array, high tariff 7-21 hours, low tariff 21-7 hours
    df = pd.DataFrame()
    # Making the dataframe be each hour of the year 2023
    df["Timestep"] = pd.date_range(start='1/1/2023', end='1/1/2024', freq='H')
    # Assigning the electricity prices for specific hours throughout the year
    # Setting a high tariff from 7:00 to 21:00
    df["Electricity_priceCHF"] = low_tariff
    df.loc[(df["Timestep"].dt.hour >= 7) & (df["Timestep"].dt.hour < 21), "Electricity_priceCHF"] = high_tariff
    # Making this a numpy array
    elec_price = df["Electricity_priceCHF"].to_numpy()
    return elec_price[:-1]

def get_iwb_tariff_data(tariff: str = 'power small'):
    """
    Function to get the simple tariff data - source: https://www.iwb.ch/servicecenter/stromtarife/aktuelle-tarife
    :param tariff: str,  'power small', 'power small plus', 'power medium' or 'power medium plus'
    :return: numpy array with the tariff data
    """
    if tariff == 'power small':
        high_tariff = 0.1239 # High tariff [CHF/kWh]
        low_tariff  = 0.1007 # Low tariff [CHF/kWh]
    elif tariff == 'power small plus':
        high_tariff = 0.1072 # High tariff [CHF/kWh]
        low_tariff  = 0.0829 # Low tariff [CHF/kWh]
    elif tariff == 'power medium':
        high_tariff = 0.1066 # High tariff [CHF/kWh]
        low_tariff  = 0.0824 # Low tariff [CHF/kWh]
    elif tariff == 'power medium plus':
        high_tariff = 0.1023 # High tariff [CHF/kWh]
        low_tariff  = 0.0786 # Low tariff [CHF/kWh]
    else:
        raise ValueError("Error: Invalid type of tariff")
    
    # Additional tariffs for the IWB tariffs
    high_network_tariff = 0.1486 + 0.0005 # Network tariff [CHF/kWh]
    low_network_tariff  = 0.0921 + 0.0005 # Network tariff [CHF/kWh]

    # Taxes in Basel and Switzerland
    high_tax = 0.0167 + 0.0520 + 0.0248 # Tax [CHF/kWh]
    low_tax  = 0.0167 + 0.0400 + 0.0248 # Tax [CHF/kWh]

    # Adding all the tariffs together
    high_tariff = high_tariff + high_network_tariff + high_tax
    low_tariff  = low_tariff + low_network_tariff + low_tax

    df = pd.DataFrame()
    # Making the dataframe be each hour of the year 2023
    df["Timestep"] = pd.date_range(start='1/1/2023', end='1/1/2024', freq='H')
    # Setting a high tariff from 6:00 to 20:00
    df["Electricity_priceCHF"] = low_tariff
    df.loc[(df["Timestep"].dt.hour >= 6) & (df["Timestep"].dt.hour < 20), "Electricity_priceCHF"] = high_tariff
    elec_price = df["Electricity_priceCHF"].to_numpy()
    return elec_price[:-1]


def get_groupe_e_tariff_data(resolution: str = 'hourly', tariff: str = 'vario_plus'):
    """
    Function to get the tariff data from the API of Groupe E. - source: https://www.groupe-e.ch/de/energie/elektrizitaet/privatkunden/vario
    :param resolution: str, 'hourly' or 'daily' or '15minutes'
    :param tariff: str, 'vario_plus' or 'vario_grid' or 'dt_plus'
    :return: numpy array with the tariff data
    """
    # Base URL of the API
    base_url = 'https://api.tariffs.groupe-e.ch'

    # Endpoint for accessing all data
    endpoint = '/v1/tariffs'

    # Full URL
    url = f"{base_url}{endpoint}"

    # Parameters for the query - accessing data from all tariffs for the year 2023
    params = {
        'start_timestamp': '2023-01-01T00:00:00+01:00',
        'end_timestamp': '2023-12-31T23:59:59+01:00'
    }

    # Making the GET request
    response = requests.get(url, params=params)

    # Checking the status of the request
    if response.status_code == 200:
        # Parsing the JSON response
        data = response.json()
        # print(data)
    else:
        print(f"Error: {response.status_code}, {response.text}")

    df = pd.DataFrame(data) # Convert start_timestamp to datetime with timezone info
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], utc=True)
    # Set start_timestamp as the index
    df = df.set_index('start_timestamp')

    if resolution == 'hourly':
        # Resample the data to hourly intervals and aggregate using mean
        resampled_df = df[f"{tariff}"].resample('H').mean()
        elec_price_MWh = resampled_df.to_numpy() # Rp/MWh
        elec_price = elec_price_MWh / 1e2 # CHF/kWh


    elif resolution == 'daily':
        # Creating a DataFrame with the tariff data
        resampled_day = df[f"{tariff}"].resample('D').mean()
        elec_price_MWh = resampled_day.to_numpy() # Rp/kWh
        elec_price = elec_price_MWh / 1e2 # CHF/kWh

    elif resolution == '15minutes':
        # Creating a DataFrame with the tariff data
        elec_price_MWh = df[f"{tariff}"].to_numpy() # Rp/kWh
        elec_price = elec_price_MWh / 1e2 # CHF/kWh
    
    return elec_price

def configure_plots(style: str = 'default', colors: str = 'vibrant'):
    """
    Function to configure the plots with the desired settings.
    :param style: str, 'default' or 'fancy'
    :param colors: str, 'vibrant' or 'diverging' or 'grayscale'
    """
    os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
    if style == 'default':
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['text.usetex'] = True

    elif style == 'fancy':
        # Define a vibrant color palette
        vibrant_colors = [
            "#BB0000",  # Red
            "#D6BA00",  # Gold
            "#002BFF",  # Blue
            "#2A9D8F",  # Teal
            "#E76F51",  # Coral
            "#F4A261",  # Orange
        ]
        diverging_colors = [
            "#1984c5", 
            "#22a7f0",
            "#63bff0", 
            "#a7d5ed", 
            "#e2e2e2", 
            "#e1a692", 
            "#de6e56", 
            "#e14b31", 
            "#c23728"
        ]
        grayscale_colors = [
            "#000000",  # Black
            "#1A1A1A",  # Very Dark Gray
            "#333333",  # Dark Gray
            "#4D4D4D",  # Darker Medium Gray
            "#666666",  # Medium Gray
            "#808080",  # Medium Light Gray
            "#999999",  # Light Gray
            "#B3B3B3",  # Light Medium Gray
            "#CCCCCC",  # Very Light Gray
            "#E6E6E6"   # Near White
        ]
        if colors == 'vibrant':
            chosen_colors = vibrant_colors
        elif colors == 'diverging':
            chosen_colors = diverging_colors
        elif colors == 'grayscale':
            chosen_colors = grayscale_colors
        else:
            chosen_colors = vibrant_colors
    
        # Update rcParams 
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
            "font.size": 12,
            "lines.color": "k",     # black lines
            "lines.linewidth": 1,   # thick lines
            "figure.figsize": [6, 4], # default fig size
            "legend.labelcolor": "black",
            "legend.edgecolor": "black",
            "legend.fancybox": False,
            "legend.facecolor": "white",
            "legend.shadow": False,
            "legend.fontsize": 10,
            "axes.labelsize": 12,
            "axes.grid": False,
            "grid.linestyle": '--',
            "grid.linewidth": 0.5,
            "grid.color": "gray",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
        })
        plt.rcParams['axes.prop_cycle'] = cycler(color=chosen_colors)

    else:
        plt.rcParams.update(plt.rcParamsDefault)

    return None

def configure_sns_plots(style: str = 'default'):
    """
    Function to configure the seaborn plots with the desired settings.
    :param style: str, 'default' or 'fancy'
    """
    os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
    if style == 'default':
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['text.usetex'] = True

    elif style == 'fancy':
        plt.rcParams.update(plt.rcParamsDefault)
        # Update rcParams 
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
            "font.size": 12,
            "lines.color": "k",     # black lines
            "lines.linewidth": 1,   # thick lines
            "figure.figsize": [6, 4], # default fig size
            "legend.labelcolor": "black",
            "legend.edgecolor": "black",
            "legend.fancybox": False,
            "legend.facecolor": "white",
            "legend.shadow": False,
            "legend.fontsize": 10,
            "axes.labelsize": 12,
            "axes.grid": False,
            "grid.linestyle": '--',
            "grid.linewidth": 0.5,
            "grid.color": "gray",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
        })
    else:
        plt.rcParams.update(plt.rcParamsDefault)

    return None

def get_electricity_profile_demand(resolution: str = 'hourly'):
    """
    Function to get the electricity demand - source: http://data.sccer-jasm.ch/demand-hourly-profile/2019-02-27/
    :param resolution: str, 'hourly'
    :return: normalised numpy array with the profile of electricity demand data for lighting and electric appliances
    """
    folder_data_demand = 'data/jasm-demand-hourly-profile-2019-02-27'

    # Load the data
    lighting_demand = 'lighting-demand-profile.csv'
    electric_appliance_demand = 'other-electric-appliances-demand-profile.csv'
    lighting_demand = pd.read_csv(os.path.join(folder_data_demand, lighting_demand))
    electric_appliance_demand = pd.read_csv(os.path.join(folder_data_demand, electric_appliance_demand))

    # Time index for year 2023 as df
    date_range = pd.date_range(start='2023-01-01', end='2023-12-31 23:00:00', freq='H')
    df = pd.DataFrame(index=date_range)

    # Months for each season
    winter = [1, 2, 3]
    spring = [4, 5, 6]
    summer = [7, 8, 9]
    autumn = [10, 11, 12]

    # Set days
    winter_days = df.index.month.isin(winter)
    spring_days = df.index.month.isin(spring)
    summer_days = df.index.month.isin(summer)
    autumn_days = df.index.month.isin(autumn)

    # Count number of days in each season
    n_winter_days = len(df.loc[winter_days].resample('D').mean())
    n_spring_days = len(df.loc[spring_days].resample('D').mean())
    n_summer_days = len(df.loc[summer_days].resample('D').mean())
    n_autumn_days = len(df.loc[autumn_days].resample('D').mean())

    # Obtaining demands from data
    winter_lighting_demand = lighting_demand['Winter (Jan)']
    spring_lighting_demand = lighting_demand['Spring (April)']
    summer_lighting_demand = lighting_demand['Summer (July)']
    autumn_lighting_demand = lighting_demand['Autumn (Oct)']
    winter_ele_app_week_demand = electric_appliance_demand['WIN-WK']
    spring_ele_app_week_demand = electric_appliance_demand['INT-WK']
    summer_ele_app_week_demand = electric_appliance_demand['SUM-WK']
    autumn_ele_app_week_demand = electric_appliance_demand['INT-WK']
    winter_ele_app_weekend_demand = electric_appliance_demand['WIN-WE']
    spring_ele_app_weekend_demand = electric_appliance_demand['INT-WE']
    summer_ele_app_weekend_demand = electric_appliance_demand['SUM-WE']
    autumn_ele_app_weekend_demand = electric_appliance_demand['INT-WE']

    # Repeate profile for each day of each season - tile on top of each other
    winter_lighting_demand = np.tile(winter_lighting_demand, n_winter_days)
    spring_lighting_demand = np.tile(spring_lighting_demand, n_spring_days)
    summer_lighting_demand = np.tile(summer_lighting_demand, n_summer_days)
    autumn_lighting_demand = np.tile(autumn_lighting_demand, n_autumn_days)

    # Assign the demand to the correct days
    df.loc[winter_days, 'lighting_demand'] = winter_lighting_demand
    df.loc[spring_days, 'lighting_demand'] = spring_lighting_demand
    df.loc[summer_days, 'lighting_demand'] = summer_lighting_demand
    df.loc[autumn_days, 'lighting_demand'] = autumn_lighting_demand

    # Define the conditions for seasons
    summer = (df.index.month >= 7) & (df.index.month <= 9)
    autumn = (df.index.month >= 10) & (df.index.month <= 12)
    winter = (df.index.month == 1) | (df.index.month <= 3)
    spring = (df.index.month >= 4) & (df.index.month <= 6)

    # Define conditions for weekdays and weekends
    weekdays = df.index.weekday < 5
    weekends = df.index.weekday >= 5

    # Tile the demand for each season and each day of the week for electric appliance demand
    summer_weekday = np.tile(summer_ele_app_week_demand, len(df[summer & weekdays]) // 24)
    summer_weekend = np.tile(summer_ele_app_weekend_demand, len(df[summer & weekends]) // 24)
    winter_weekday = np.tile(winter_ele_app_week_demand, len(df[winter & weekdays]) // 24)
    winter_weekend = np.tile(winter_ele_app_weekend_demand, len(df[winter & weekends]) // 24)
    autumn_weekday = np.tile(autumn_ele_app_week_demand, len(df[autumn & weekdays]) // 24)
    autumn_weekend = np.tile(autumn_ele_app_weekend_demand, len(df[autumn & weekends]) // 24)
    spring_weekday = np.tile(spring_ele_app_week_demand, len(df[spring & weekdays]) // 24)
    spring_weekend = np.tile(spring_ele_app_weekend_demand, len(df[spring & weekends]) // 24)

    # Assign the demand to the correct days
    df.loc[summer & weekdays, 'ele_app_demand'] = summer_weekday
    df.loc[summer & weekends, 'ele_app_demand'] = summer_weekend
    df.loc[winter & weekdays, 'ele_app_demand'] = winter_weekday
    df.loc[winter & weekends, 'ele_app_demand'] = winter_weekend
    df.loc[autumn & weekdays, 'ele_app_demand'] = autumn_weekday
    df.loc[autumn & weekends, 'ele_app_demand'] = autumn_weekend
    df.loc[spring & weekdays, 'ele_app_demand'] = spring_weekday
    df.loc[spring & weekends, 'ele_app_demand'] = spring_weekend

    # Normalise each demand so that it adds to 1
    df['lighting_demand'] = df['lighting_demand'] / df['lighting_demand'].sum()
    df['ele_app_demand'] = df['ele_app_demand'] / df['ele_app_demand'].sum()

    # Convert to numpy arrays
    lighting_profile_demand = df['lighting_demand'].to_numpy()
    electric_appliance__profile_demand = df['ele_app_demand'].to_numpy()

    # Return the normalised demands
    return lighting_profile_demand, electric_appliance__profile_demand

def get_solar_irradiance_data():
    """
    Function to get the solar irradiance data - source: 
    :return: numpy array with the solar irradiance data
    """
    data = pd.read_csv("data/data_bernardino_2022.csv")                     # Import data
    return data["SolFlat"].to_numpy()                            # Solar irradiance [kWh/m^2]


def get_PWA_lines(x_vals, y_vals):
    """
    Function to get the PWA lines to add in constraints. 
    """
    # Calculate slopes (m) between consecutive points
    slopes = np.diff(y_vals) / np.diff(x_vals)  
    intercepts = y_vals[:-1] - slopes * x_vals[:-1]
    
    return slopes, intercepts