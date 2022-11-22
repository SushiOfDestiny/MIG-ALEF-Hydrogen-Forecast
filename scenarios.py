import numpy as np
import pandas as pd
import tech_eco_data


nHours = 8760
t = np.arange(1,nHours + 1)

zones = ['PACA']

yearZero = 2020
yearFinal = 2050
yearStep = 10
yearList = [yr for yr in range(yearZero, yearFinal+yearStep, yearStep)] # +1 to include the final year
nYears = len(yearList)

scenario = {}

scenario['resourceDemand'] =  pd.concat(
    (
        pd.DataFrame(data = { 
              'YEAR': year, 
              'TIMESTAMP': t, # We add the TIMESTAMP so that it can be used as an index later. 
              'electricity': np.zeros(nHours),
              'hydrogen': 360 * (1 + .025) ** (k * yearStep), # Hourly constant but increasing demand
              'gas': np.zeros(nHours), 
              'uranium': np.zeros(nHours)
             } 
        ) for k, year in enumerate(yearList) 
    ) 
) 

scenario['conversionTechs'] = [] 
for k, year in enumerate(yearList): 
    tech = "Offshore wind - floating"
    maxcap = 10000
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year) 
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minCapacity': 0,'maxCapacity': maxcap, 
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'hydrogen':0},
                'EnergyNbhourCap': 0, # used for hydroelectricity 
                'capacityLim': 100e3, } 
            }
         )
    )

    tech = "Onshore wind"
    maxcap = 10000
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year) 
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minCapacity': 0,'maxCapacity': maxcap, 
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'hydrogen':0},
                'EnergyNbhourCap': 0, # used for hydroelectricity 
                'capacityLim': 100e3, 
                }, 
            }
         )
    )

    tech = "Ground PV"
    maxcap = 10000
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year) 
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minCapacity': 0,'maxCapacity': maxcap, 
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'hydrogen':0},
                'EnergyNbhourCap': 0, # used for hydroelectricity 
                'capacityLim': 100e3, 
                }, 
            }
         )
    )

    tech = "Electrolysis"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year) 
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minCapacity': 0,'maxCapacity': 100e3, 
                'EmissionCO2': 0, 'Conversion': {'electricity': -1, 'hydrogen':0.65},
                'EnergyNbhourCap': 0, # used for hydroelectricity 
                'capacityLim': 100e3, 
                }, 
            }
         )
    )

    tech = "SMR"
    capex, opex, lifespan = 800e3, 40e3, 60
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minCapacity': 0,'maxCapacity': 100e3, 
                'EmissionCO2': 0, 'Conversion': {'electricity': 0, 'hydrogen': 1, 'gas': -1.43},
                'EnergyNbhourCap': 0, # used for hydroelectricity 
                'capacityLim': 100e3, 
                }, 
            }
         )
    )

    tech = "Existing SMR"
    capex, opex, lifespan = 0e3, 40e3, 30
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minCapacity': 320 if year == yearZero else 0,'maxCapacity': 320 if year == yearZero  else 0, 
                'EmissionCO2': 0, 'Conversion': {'electricity': 0, 'hydrogen': 1, 'gas': -1.43},
                'EnergyNbhourCap': 0, # used for hydroelectricity 
                'capacityLim': 320 if year == yearZero else 0, 
                }, 
            }
         )
    )

    tech = "SMR + CCS1"
    capex, opex, lifespan = 900e3, 45e3, 60
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minCapacity': 0,'maxCapacity': 100e3, 
                'EmissionCO2': -169, 'Conversion': {'electricity': -0.17, 'hydrogen': 1, 'gas': -1.43},
                'EnergyNbhourCap': 0, # used for hydroelectricity 
                'capacityLim': 100e3, 
                }, 
            }
         )
    )

    tech = "SMR + CCS2"
    capex, opex, lifespan = 1000e3, 50e3, 60
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minCapacity': 0,'maxCapacity': 100e3, 
                'EmissionCO2': -268, 'Conversion': {'electricity': -0.34, 'hydrogen': 1, 'gas': -1.43},
                'EnergyNbhourCap': 0, # used for hydroelectricity 
                'capacityLim': 100e3, 
                }, 
            }
         )
    )

    tech = "CCS1"
    capex, opex, lifespan = 100e3, 0e3, 60
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Carbon capture',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 'capacityLim': 100e3}, 
            }
         )
    )

    tech = "CCS2"
    capex, opex, lifespan = 100e3, 0e3, 60
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Carbon capture',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 'capacityLim': 100e3, }, 
            }
         )
    )

scenario['conversionTechs'] =  pd.concat(scenario['conversionTechs'], axis=1) 

scenario['storageTechs'] = [] 
for k, year in enumerate(yearList): 
    tech = "Battery"
    capex1, opex1, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 1h', hyp='ref', year=year)
    capex4, opex4, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 4h', hyp='ref', year=year)
    capex_per_kWh = (capex4 - capex1) / 3
    capex_per_kW = capex1 - capex_per_kWh

    scenario['storageTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'resource': 'electricity',
                'storagelifeSpan': lifespan, 
                'storagePowerCost': capex_per_kW, 
                'storageEnergyCost': capex_per_kWh, 
                'storageOperationCost': opex1, # TODO: according to RTE OPEX seems to vary with energy rather than power
                'p_max': 5000, 
                'c_max': 50000, 
                'chargeFactors': {'electricity': 0.9200},
                'dischargeFactors': {'electricity': 1.09},
                'dissipation': 0.0085,
                }, 
            }
         )
    )

    tech = "Salt cavern"
    scenario['storageTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 
               'resource': 'hydrogen', 
               'storagelifeSpan': 40, 
                'storagePowerCost': 0, 
                'storageEnergyCost': 350e3, 
                'storageOperationCost': 2e3, 
                'p_max': 10000, 
                'c_max': 100000, 
                'chargeFactors': {'electricity': 0.0168, 'hydrogen': 1.0},
                'dischargeFactors': {'hydrogen': 1.0},
                'dissipation': 0,
                }, 
            }
         )
    )

scenario['storageTechs'] =  pd.concat(scenario['storageTechs'], axis=1) 

scenario['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.165, nYears),
    index=yearList, columns=('carbonTax',))

scenario['carbonGoals'] = pd.DataFrame(data=np.linspace(974e6, 205e6, nYears),
    index=yearList, columns=('carbonGoals',))

scenario['maxBiogasCap'] = pd.DataFrame(data=np.linspace(0, 310e6, nYears),
    index=yearList, columns=('maxBiogasCap',))

scenario['gridConnection'] = pd.read_csv("Data\Raw\CalendrierHPHC_TIME.csv", sep=',', decimal='.', skiprows=0,
                                comment="#").set_index(["TIMESTAMP"])

scenario['economicParameters'] = pd.DataFrame({
    'discountRate':[0.04], 
    'financeRate': [0.04]
    }
)

df_res_ref = pd.read_csv('./Data/Raw/set2020-2050_horaire_TIMExRESxYEAR.csv', 
    sep=',', decimal='.', skiprows=0,comment="#").set_index(["YEAR", "TIMESTAMP",'RESOURCES'])

scenario['resourceImportPrices'] = pd.concat(
    (
        pd.DataFrame(data={
            'YEAR': year, 
            'TIMESTAMP': t, 
            'electricity': df_res_ref.loc[(year, slice(None), 'electricity'),'importCost'].values,
            'natural gas': 2 * df_res_ref.loc[(year, slice(None), 'gazNat'),'importCost'].values,
            'biogas': 150 * np.ones(nHours),
            'uranium': 2.2 * np.ones(nHours),
            'hydrogen': 6/33 * 1000 * np.ones(nHours),
        }) for k, year in enumerate(yearList[1:])
    )
)

scenario['resourceImportCO2eq'] = pd.concat(
    (
        pd.DataFrame(data={
            'YEAR': year, 
            'TIMESTAMP': t, 
            'electricity': df_res_ref.loc[(year, slice(None), 'electricity'),'emission'].values,
            'gas': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1 + 203.5  * (1 - tech_eco_data.get_biogas_share_in_network_RTE(year)), # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050. 
            'natural gas': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1 + 203.5  * (1 - tech_eco_data.get_biogas_share_in_network_RTE(year)), # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050. 
            'biogas': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1,
            'uranium': 0 * np.ones(nHours),
            'hydrogen': max(0, 0.05  - .03 * (year - yearZero)/(2050 - yearZero)) * 11 / 33, # Taking 100 yr GWP of H2 and 5% losses due to upstream leaks. Leaks fall to 2% in 2050 See: https://www.energypolicy.columbia.edu/research/commentary/hydrogen-leakage-potential-risk-hydrogen-economy
        }) for k, year in enumerate(yearList[1:])
    )
)

scenario['convTechList'] = ["Offshore wind - floating", "Onshore wind", "Ground PV", "Electrolysis"]
ctechs = scenario['convTechList']
availabilityFactor = pd.read_csv('Data/Raw/availabilityFactor2020-2050_PACA_TIMExTECHxYEAR - renamed.csv',
                                 sep=',', decimal='.', skiprows=0).set_index(["YEAR", "TIMESTAMP", "TECHNOLOGIES"])
itechs = availabilityFactor.index.isin(ctechs, level=2)
scenario['availability'] = availabilityFactor.loc[(slice(None), slice(None), itechs)]

scenario["yearList"] = yearList 
scenario["transitionFactors"] =pd.DataFrame(
    {'TECHNO1':['Existing SMR', 'Existing SMR', 'SMR', 'SMR', 'SMR + CCS1'],
    'TECHNO2':['SMR + CCS1','SMR + CCS2', 'SMR + CCS1','SMR + CCS2','SMR + CCS2'],
    'TransFactor': 1}).set_index(['TECHNO1','TECHNO2'])