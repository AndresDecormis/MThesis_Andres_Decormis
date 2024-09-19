
DATAPACKAGE: DEMAND HOURLY PROFILE IN STEM
===========================================================================

Package Version: 2019-02-27

Space heating, process heat, water heating, air conditioning and other
electric appliances (e.g. refrigerators, ovens, information&communication
technologies, washers and dryers,etc.) hourly profile by typical day and
sector - Assumption in STEM model



We follow the Data Package standard by the Frictionless Data project, a
part of the Open Knowledge Foundation: http://frictionlessdata.io/



License and attribution
===========================================================================

Attribution:
    


Version history
===========================================================================

* 2019-02-27 Not documented


Resources
===========================================================================

* [Package description page](http://data.sccer-jasm.ch/demand-hourly-profile/2019-02-27/)


Sources
===========================================================================

* [Vögelin P., Panos E., Buffat R., Becutti G., Kannan R., et al. (2016). System modelling for assessing the potential of decentralised biomass-CHP plants to stabilise the Swiss electricity network with increased fluctuating renewable generation, Final report of the Swiss Federal Office of Energy. ](www.bfe.admin.ch/php/modules/enet/streamfile.php?file=000000011337.pdf)
* [R. Kannan and H. Turton (2014). Switzerland energy transition scenarios – Development and application of the Swiss TIMES Energy system Model (STEM), Final Report to Swiss Federal Office of Energy, Bern](https://www.psi.ch/eem/PublicationsTabelle/2014-STEM-PSI-Bericht-14-06.pdf)
* [Lighting: Stokes M., Rylatt M. and Lomas K. (2004) A simple model of domestic lighting demand, Energy and Buildings 36 (2) pp 103-116](https://www.dora.dmu.ac.uk/handle/2086/8963)


Field documentation
===========================================================================

space-heating-demand-profile.csv
---------------------------------------------------------------------------

* Sector
    - Type: string
    - Description: Industry, residential (New/Existing Single/Multi Family Houses), services
* Hour
    - Type: number
    - Description: Hour of the typical day
* WIN-WK
    - Type: number
    - Description: Demand in a typical winter working day (% of the total of the year)
* INT-WK
    - Type: number
    - Description: Demand in a typical intermediate season working day (% of the total of the year)
* SUM-WK
    - Type: number
    - Description: Demand in a typical summer working day (% of the total of the year)
* WIN-WE
    - Type: number
    - Description: Demand in a typical winter weekend (% of the total of the year)
* INT-WE
    - Type: number
    - Description: Demand in a typical intermediate season weekend (% of the total of the year)
* SUM-WE
    - Type: number
    - Description: Demand in a typical summer weekend (% of the total of the year)


water-heating-demand-profile.csv
---------------------------------------------------------------------------

* Sector
    - Type: string
    - Description: Industry, residential (New/Existing Single/Multi Family Houses), services
* Hour
    - Type: number
    - Description: Hour of the typical day
* WIN-WK
    - Type: number
    - Description: Demand in a typical winter working day (% of the total of the year)
* INT-WK
    - Type: number
    - Description: Demand in a typical intermediate season working day (% of the total of the year)
* SUM-WK
    - Type: number
    - Description: Demand in a typical summer working day (% of the total of the year)
* WIN-WE
    - Type: number
    - Description: Demand in a typical winter weekend (% of the total of the year)
* INT-WE
    - Type: number
    - Description: Demand in a typical intermediate season weekend (% of the total of the year)
* SUM-WE
    - Type: number
    - Description: Demand in a typical summer weekend (% of the total of the year)


process-heat-demand-profile.csv
---------------------------------------------------------------------------

* Sector
    - Type: string
    - Description: Industry
* Hour
    - Type: number
    - Description: Hour of the typical day
* WIN-WK
    - Type: number
    - Description: Demand in a typical winter working day (% of the total of the year)
* INT-WK
    - Type: number
    - Description: Demand in a typical intermediate season working day (% of the total of the year)
* SUM-WK
    - Type: number
    - Description: Demand in a typical summer working day (% of the total of the year)
* WIN-WE
    - Type: number
    - Description: Demand in a typical winter weekend (% of the total of the year)
* INT-WE
    - Type: number
    - Description: Demand in a typical intermediate season weekend (% of the total of the year)
* SUM-WE
    - Type: number
    - Description: Demand in a typical summer weekend (% of the total of the year)


air-conditioning-demand-profile.csv
---------------------------------------------------------------------------

* Sector
    - Type: string
    - Description: Industry and services, residential
* Hour
    - Type: number
    - Description: Hour of the typical day
* WIN-WK
    - Type: number
    - Description: Demand in a typical winter working day (% of the total of the year)
* INT-WK
    - Type: number
    - Description: Demand in a typical intermediate season working day (% of the total of the year)
* SUM-WK
    - Type: number
    - Description: Demand in a typical summer working day (% of the total of the year)
* WIN-WE
    - Type: number
    - Description: Demand in a typical winter weekend (% of the total of the year)
* INT-WE
    - Type: number
    - Description: Demand in a typical intermediate season weekend (% of the total of the year)
* SUM-WE
    - Type: number
    - Description: Demand in a typical summer weekend (% of the total of the year)


other-electric-appliances-demand-profile.csv
---------------------------------------------------------------------------

* Sector
    - Type: string
    - Description: Residential
* Hour
    - Type: number
    - Description: Hour of the typical day
* WIN-WK
    - Type: number
    - Description: Demand in a typical winter working day (% of the total of the year)
* INT-WK
    - Type: number
    - Description: Demand in a typical intermediate season working day (% of the total of the year)
* SUM-WK
    - Type: number
    - Description: Demand in a typical summer working day (% of the total of the year)
* WIN-WE
    - Type: number
    - Description: Demand in a typical winter weekend (% of the total of the year)
* INT-WE
    - Type: number
    - Description: Demand in a typical intermediate season weekend (% of the total of the year)
* SUM-WE
    - Type: number
    - Description: Demand in a typical summer weekend (% of the total of the year)


lighting-demand-profile.csv
---------------------------------------------------------------------------

* Hour
    - Type: number
    - Description: Hour of the day (% of the total of the year)
* Winter (Jan)
    - Type: number
    - Description: Profile during the months in winter (% of the total of the year)
* Summer (July)
    - Type: number
    - Description: Profile during the months in summer (% of the total of the year)
* Spring (April)
    - Type: number
    - Description: Profile during the months in spring (% of the total of the year)
* Autumn (Oct)
    - Type: number
    - Description: Profile during the months in autumn (% of the total of the year)


Feedback
===========================================================================

Thank you for using data provided by SCCER JA S&M. If you have
any question or feedback, please do not hesitate to contact us.

For this data package, contact:
 <evangelos.panos@psi.ch>

 <kannan.ramachandran@psi.ch>

For general issues, find our team contact details on our website:
http://www.sccer-jasm.ch














