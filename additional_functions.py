# All the functions that are used in the main code and not in the main code itself are stored here.

import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from cycler import cycler

def get_hydrogen_price():
    """
    Function to get the hydrogen price - source: https://data.sccer-jasm.ch/import-prices/2020-08-01/
    :return: float with the hydrogen price
    """
    hydrogen_market_price = 3.3 # Hydrogen market price [CHF/kgH2] - 3.3: 
    return hydrogen_market_price       # Hydrogen price [CHF/kg]

def get_spot_prices():
    """
    Function to get the spot prices - source https://transparency.entsoe.eu/
    :return: numpy array with the spot prices
    """
    data = pd.read_csv("data/data2023.csv")                     # Import data
    data["Electricity_priceCHF"] = data["Electricity_priceCHF"] * 1e-3 # Convert to CHF/kWh
    return data["Electricity_priceCHF"].to_numpy()       # Electricity price [CHF/kWh]

def get_temperature_data():
    """
    Function to get the temperature data - source: https://opendata.swiss/en/dataset/stundlich-aktualisierte-meteodaten-seit-1992
    :return: numpy array with the temperature data
    """
    data = pd.read_csv("data/data2023.csv")                 # Import data
    return data["TemperatureC"].to_numpy()       # Temperature [°C]

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