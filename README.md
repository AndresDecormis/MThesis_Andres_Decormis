# MThesis_Andres_Decormis

This README file refers to the Pyhton scripts developed and used for the master thesis:  

" Optimal Agent-Based Coordination of Energy Communities:
A Case Study of Waste Heat Recovery from Hydrogen Technologies" 

The repository containing this README file contains the following documents (with a brief descrpition):

- centralised_optimisation.py: 
		Script that obtains the solution to minimise total operational costs of a multi-energy system in a centralised optimisation. 
- distributed_coordination_ADMM.py:
		Script that obtains the solution to minimise total operational costs of a multi-energy community with the Alternating Direction Method of Multipliers (ADMM) 
- focs_shapley_quantification.py:
		Script that calculates the Fair Operational Cost Savings (FOCS) value of energy technologies in a  multi-energy system. 
- additional_functions.py:
		Various helpful functions that perform different tasks, for example: retrieve data from csv files into DataFrames or arrays,  assign values based on run configurations, configure plot style, etc.
- parameters.py:
		Contains all the technoeconomic parameters and boundary condition values used for the optimisation runs.
- MScThesis_Final_Report_Decormis_Andres.pdf:
		The final report from this master thesis.
		
This repository also contains the following folders:
- ~/data: this folder contains a set of csv files that contain hourly data which is called python functions for the optimisation problem. 
- ~/results: this folder stores the results obtained from running the python scripts described before. It already contains some of the results which are included in the thesis
- ~/post_processing_functions: this folder contains a number of python scripts that we used to postprocess some of the results. These scripts are not performing calculations but provided here in case they might be useful. These scripts might need modifications before running them. 
- ~/literature: this folder contains various documents (articles, reports, etc.) which were useful as literature review for this thesis.
		

The following packages need to be installed in the python environment used to run the python scripts in this repository:
- cvxpy 1.5.2
- gurobi 11.02
- numpy 1.25.0
- pandas 1.5.3
- matplotlib 3.9.1
- itertools
- functools 
- cycler

To make the plots with LaTex typesetting Texlive 2024 from MacTex is also installed. 