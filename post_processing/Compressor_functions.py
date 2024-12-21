# Script to view compressor functions
import numpy as np
import matplotlib.pyplot as plt
import additional_functions as af
from parametersv2 import *
import os

af.configure_plots(style='fancy')

save_figures            = True              # Save the figures
file_path               = "results/00-compressor_functions"
# Create compressor_functions folder if it does not exist
if not os.path.exists(file_path):
    os.makedirs(file_path)

hydrogen_values                        = np.linspace(0.1, 20, 1000)
high_pressure_values                   = np.arange(30, 900, 10)

# Define experimental results for energy consumption of the compressor
timo_900bar = 2.5470            # Energy consumption of the compressor at 900 bar [kWh/kg]
energy_con_timo_900bar = timo_900bar * hydrogen_values  # [kWh]

# Functions to calculate the compressor coefficients
def compressor_adiabatic(T_in, eff_co, P_out, P_in, M_H2, GAMMA):
    """
    Function to calculate the adiabatic compression coefficient of the compressor
    """
    comp_ratio  = (P_out / P_in)    # Pressure ratio
    return (((R/M_H2)*T_in) * (GAMMA /(GAMMA-1)) * ((comp_ratio)**((GAMMA-1)/GAMMA)-1) / 3600) / eff_co # [kWh/kg]
def compressor_isothermal(T_in, eff_co, P_out, P_in, M_H2):
    """
    Function to calculate the isothermal compression coefficient of the compressor
    """
    comp_ratio  = (P_out / P_in)    # Pressure ratio
    return R*T_in*np.log(comp_ratio) / 3600 / M_H2 / eff_co                  # [kWh/kg]

# Calculate the adiabatic and isothermal compression coefficients for each value of the hydrogen input and pressure values
adiabatic_values    = np.zeros(len(high_pressure_values))
isothermal_values   = np.zeros(len(high_pressure_values))

for j in range(len(high_pressure_values)):
    adiabatic_values[j]   = compressor_adiabatic(T_in, eff_co, high_pressure_values[j], P_in, M_H2, GAMMA)
    isothermal_values[j]  = compressor_isothermal(T_in, eff_co, high_pressure_values[j], P_in, M_H2)

energy_consumption_adiabatic  = np.zeros((len(hydrogen_values),len(high_pressure_values)))
energy_consumption_isothermal = np.zeros((len(hydrogen_values),len(high_pressure_values)))

for i in range(len(hydrogen_values)):
    for j in range(len(high_pressure_values)):
        energy_consumption_adiabatic[i,j] = adiabatic_values[j] * hydrogen_values[i]
        energy_consumption_isothermal[i,j] = isothermal_values[j] * hydrogen_values[i]

# Plot the energy consumption of the compressor as a 3d surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(high_pressure_values, hydrogen_values)
ax.plot_surface(X, Y, energy_consumption_adiabatic, cmap='viridis')
ax.plot_surface(X, Y, energy_consumption_isothermal, cmap='plasma')
ax.set_xlabel('High pressure [bar]')
ax.set_ylabel('Hydrogen input [kg]')
ax.set_zlabel('Energy consumption [kWh]')
ax.set_title('Energy consumption of the compressor')

if save_figures:
    plt.savefig(f"{file_path}/energy_consumption_compressor.pdf", format="pdf", bbox_inches="tight")

# Plot the energy consumption of the compressor for a high pressure of 440 bar as a function of the hydrogen input
plt.figure()
pressure_index = np.argmin(np.abs(high_pressure_values - 440))
plt.plot(hydrogen_values, energy_consumption_adiabatic[:, pressure_index], label='Adiabatic')
plt.plot(hydrogen_values, energy_consumption_isothermal[:, pressure_index], label='Isothermal')
plt.xlabel('Hydrogen input [kg]')
plt.ylabel('Energy consumption [kWh]')
plt.title('Energy consumption of the compressor at 440 bar')
plt.legend()

if save_figures:
    plt.savefig(f"{file_path}/energy_consumption_compressor_440bar.pdf", format="pdf", bbox_inches="tight")

# Plot the energy consumption of the compressor for a high pressure of 900 bar as a function of the hydrogen input
plt.figure()
pressure_index = np.argmin(np.abs(high_pressure_values - 900))
plt.plot(hydrogen_values, energy_consumption_adiabatic[:, pressure_index], label='Adiabatic')
plt.plot(hydrogen_values, energy_consumption_isothermal[:, pressure_index], label='Isothermal')
plt.plot(hydrogen_values, energy_con_timo_900bar, label='Experimental', linestyle='--')
plt.xlabel('Hydrogen input [kg]')
plt.ylabel('Energy consumption [kWh]')
plt.title('Energy consumption of the compressor at 900 bar')
plt.legend()

if save_figures:
    plt.savefig(f"{file_path}/energy_consumption_compressor_900bar.pdf", format="pdf", bbox_inches="tight")


# Plot the energy consumption of the compressor for a hydrogen input of 10 kg as a function of the high pressure
plt.figure()
hydrogen_index = np.argmin(np.abs(hydrogen_values - 10))
plt.plot(high_pressure_values, energy_consumption_adiabatic[hydrogen_index, :], label='Adiabatic')
plt.plot(high_pressure_values, energy_consumption_isothermal[hydrogen_index, :], label='Isothermal')
plt.xlabel(r'Output pressure $P_{\mathrm{out}}$[bar]')
plt.ylabel('Energy consumption [kWh]')
plt.title('Energy consumption of the compressor with 10 kg of hydrogen')
plt.legend()
if save_figures:
    plt.savefig(f"{file_path}/energy_consumption_10kg.pdf", format="pdf", bbox_inches="tight")


# Plot the energy consumption of the compressor for a hydrogen input of 10 kg as a function of the high pressure
plt.figure()
hydrogen_index = np.argmin(np.abs(hydrogen_values - 10))
plt.plot(high_pressure_values, energy_consumption_adiabatic[hydrogen_index, :], label='Adiabatic')
plt.plot(high_pressure_values, energy_consumption_isothermal[hydrogen_index, :], label='Isothermal')
plt.xlabel(r'Output pressure, $P_{\mathrm{out}}$ [bar]')
plt.ylabel(r'Compressor power, $P_{\mathrm{CO}}$ [kW]')
# plt.title('Energy consumption of the compressor with 10 kg of hydrogen')
plt.legend()
if save_figures:
    plt.savefig(f"{file_path}/compressor_power_at10kgperhour.pdf", format="pdf", bbox_inches="tight")

plt.show()