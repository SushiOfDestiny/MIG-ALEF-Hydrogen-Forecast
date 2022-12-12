##########
# RESUME #
##########
# algo doit optimiser avec effet d'échelles (NON LINEAIRE, MILP)
# 3 tailles d'électrolyseurs
# 3 tailles de pipeline


import numpy as np
import pandas as pd
import tech_eco_data


nHours = 8760

timeStep = 100  # For now only integers work
t = np.arange(1, nHours + 1, timeStep)
nHours = len(t)

zones = ['PACA']
scenar = 1

yearZero = 2020
yearFinal = 2050
yearStep = 10
# +1 to include the final year
yearList = [yr for yr in range(yearZero, yearFinal+yearStep, yearStep)]
nYears = len(yearList)
areaList = ["Nice", "Marseille", "Alpin"]


scenario = {}
scenario['areaList'] = areaList
scenario['timeStep'] = timeStep
scenario['lastTime'] = t[-1]

dist = {"Nice": {"Marseille": 200, "Alpin": 200, "Nice": 0}, "Marseille": {"Nice": 200,
                                                                           "Alpin": 200, "Marseille": 0}, "Alpin": {"Nice": 200, "Marseille": 200, "Alpin": 0}}
scenario['distances'] = pd.concat(
    (
        pd.DataFrame(data={
            'area1': area1,
            'area2': area2,
            'distances': dist[area1][area2]
        }, index=(area1, area2)
        ) for area1 in areaList
        for area2 in areaList
    )
)
scenario['distances'] = scenario['distances'].reset_index().drop_duplicates(
    subset=['area1', 'area2']).set_index(['area1', 'area2']).drop(columns='index')

# donne la liste des couples de noeuds
couples_noeuds = list(scenario['distances'].index)


def demande_h_area(scenar, area, k):
    """returns hydrogen yearly demand of an area in MWh"""

    if scenar == 0:
        # demande  annuelle en millions de kilos d'hydrogènes
        demande_t_an = [100, 150, 175, 200]
    elif scenar == 1:
        demande_t_an = [100, 739, 1092, 1974] # modifié selon scénario 1
    elif scenar == 2:
        demande_t_an = [100, 464, 848, 1647] # modifié selon scénario 2
    elif scenar == 3:
        demande_t_an = [100, 248, 239, 236]

    # tab numpy pour broadcast
    demande_t_an = np.array(demande_t_an)
    # on ramène en kg par heure
    demande_t_an = demande_t_an * 1.e6 / 8760
    # on convertit en MWh
    # on prend 33 kWh/kg (33e-3 MWh/kg) comme densité énergétique de l'hydrogène gazeux
    demande_t_an = demande_t_an * 33.e-3

    if area == "Nice":
        return 0.2 * demande_t_an[k] * np.ones(nHours)  # 20% de la demande
    elif area == "Alpin":
        return 0.1 * demande_t_an[k] * np.ones(nHours)  # 10% de la demande
    else:
        return 0.7 * demande_t_an[k] * np.ones(nHours)  # 70% de la demande


def stockage_h_area(area):
    if area == "Fos":
        return 100000
    else:
        return 0


scenario['resourceDemand'] = pd.concat(
    (
        pd.DataFrame(data={
            'AREA': area,
            'YEAR': year,
            # We add the TIMESTAMP so that it can be used as an index later.
            'TIMESTAMP': t,
            'electricity': np.zeros(nHours),
            # Hourly constant but increasing demand
            'hydrogen': demande_h_area(scenar, area, k),
            'gas': np.zeros(nHours),
        }
        ) for k, year in enumerate(yearList)
        for area in areaList
    )
)
'''
print(scenario['resourceDemand'])
print(scenario['resourceDemand'].head())
print(scenario['resourceDemand'].tail())
'''
scenario['conversionTechs'] = []

#############################
# PROPRIETES TECHNOLOGIQUES #
#############################
# un effort est fait pour reprendre les données choisies dans les scénarios du groupe infrastructure
# capex, opex, lifespan ne sont pas changés et restent déterminés par electrolyser_capex_Reksten2022

# on distingue la puissance de consommation et de production

# propriétés électrolyseurs
# facteur conversion électricité -> hydrogène
conv_el_h = 0.65 
# hydrogène produit par électrolyseur de taille S (en MW), selon scénario 1 
# (1MW de conso électrique pour 18kg/h ~ 600kW d'hydrogène produit)
power_S = 600e-3  #MW

# liste des puissances max de ressources 
# - pour une tech : produites par une installation (en MW)
# - pour une ttech : transportables par km (en MW/km)
p_max_fonc = {
    "Offshore wind - floating" : 12,
    "Onshore wind" : 6,
    "Ground PV" : 6.4*10**(-5),
    "ElectrolysisS" : power_S,
    "ElectrolysisM" : 10 * power_S,
    "ElectrolysisL" : 100 * power_S,
    "Pipeline_S" : 100, # puissance maximale de fonctionnement du pipeline (=débit max), fixée
    "Pipeline_M" : 1000,
    "Pipeline_L" : 10000,
    # capacité d'un camion : C = 600kg = 600*33 = 19800 kWh = 19.8 MWh
    # c'est aussi la qté d'hydrogène qu'un camion transport en 1 heure sur 1 km
    "truckTransportingHydrogen" : 19.8
}


for area in areaList:
    for k, year in enumerate(yearList):
        tech = "Offshore wind - floating"
        maxcap = 10000
        if area == "Alpin":
            maxcap = 0

        capex, opex, LifeSpan = tech_eco_data.get_capex_new_tech_RTE(
            tech, hyp='ref', year=year)
        scenario['conversionTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Electricity production',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                                'minCapacity': 0, 'maxCapacity': maxcap,
                                'EmissionCO2': 0, 'Conversion': {'electricity': 0.4, 'hydrogen': 0},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'capacityLim': 100e3,  # capacité de production max d'une zone et d'une techno
                                'techUnitPower': p_max_fonc[tech]  # puissance fonctionnelle maximale produite par une unité
                                }
                               }
                         )
        )

        tech = "Onshore wind"
        maxcap = 10000
        capex, opex, LifeSpan = tech_eco_data.get_capex_new_tech_RTE(
            tech, hyp='ref', year=year)
        if area == "Nice":
            capex *= 1.5
        scenario['conversionTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Electricity production',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                                'minCapacity': 0, 'maxCapacity': maxcap,
                                'EmissionCO2': 0, 'Conversion': {'electricity': 0.25, 'hydrogen': 0},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'capacityLim': 100e3, 'techUnitPower': p_max_fonc[tech]
                                },
                               }
                         )
        )

        tech = "Ground PV"
        maxcap = 10000
        capex, opex, LifeSpan = tech_eco_data.get_capex_new_tech_RTE(
            tech, hyp='ref', year=year)
        if area == "Nice":
            capex *= 2
        scenario['conversionTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Electricity production',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                                'minCapacity': 0, 'maxCapacity': maxcap,
                                'EmissionCO2': 0, 'Conversion': {'electricity': 0.16, 'hydrogen': 0},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'capacityLim': 100e3, 'techUnitPower': p_max_fonc[tech]
                                },
                               }
                         )
        )


        tech = "ElectrolysisS"
        capex, opex, LifeSpan = tech_eco_data.get_capex_new_tech_RTE(
            tech, hyp='ref', year=year)
        scenario['conversionTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Hydrogen production',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                                'minCapacity': 0, 'maxCapacity': 10000,  # cap à investir
                                'EmissionCO2': 0, 'Conversion': {'electricity': -1, 'hydrogen': conv_el_h},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'capacityLim': 100e3, 'techUnitPower': p_max_fonc[tech]
                                },
                               }
                         )
        )

        tech = "ElectrolysisM"
        capex, opex, LifeSpan = tech_eco_data.get_capex_new_tech_RTE(
            tech, hyp='ref', year=year)
        scenario['conversionTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Hydrogen production',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                                'minCapacity': 0, 'maxCapacity': 10000,
                                'EmissionCO2': 0, 'Conversion': {'electricity': -1, 'hydrogen': conv_el_h},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'capacityLim': 100e3, 'techUnitPower': p_max_fonc[tech]
                                },
                               }
                         )
        )

        tech = "ElectrolysisL"
        capex, opex, LifeSpan = tech_eco_data.get_capex_new_tech_RTE(
            tech, hyp='ref', year=year)
        scenario['conversionTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Hydrogen production',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                                'minCapacity': 0, 'maxCapacity': 10000,
                                'EmissionCO2': 0, 'Conversion': {'electricity': -1, 'hydrogen': conv_el_h},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'capacityLim': 100e3, 'techUnitPower': p_max_fonc[tech]
                                },
                               }
                         )
        )

        tech = "SMR"
        capex, opex, LifeSpan = 800e3, 40e3, 60
        scenario['conversionTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Hydrogen production',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                                'minCapacity': 0, 'maxCapacity': 100e3,
                                'EmissionCO2': 0, 'Conversion': {'electricity': 0, 'hydrogen': 1, 'gas': -1.43},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'capacityLim': 100e3, 'techUnitPower': 320
                                },
                               }
                         )
        )

        tech = "Existing SMR"
        capex, opex, LifeSpan = 0e3, 40e3, 30
        scenario['conversionTechs'].append(

            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Hydrogen production',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                                'minCapacity': 1 if (year == yearZero and area == 'Marseille') else 0, 'maxCapacity': 1 if (year == yearZero and area == 'Marseille') else 0,
                                'EmissionCO2': 0, 'Conversion': {'electricity': 0, 'hydrogen': 1, 'gas': -1.43},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'capacityLim': 320 if (year == yearZero and area == 'Marseille') else 0, 'techUnitPower': 320
                                },
                               }
                         )
        )

        tech = "SMR + CCS1"
        capex, opex, LifeSpan = 900e3, 45e3, 60
        scenario['conversionTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Hydrogen production',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                                'minCapacity': 0, 'maxCapacity': 100e3,
                                'EmissionCO2': -169, 'Conversion': {'electricity': -0.17, 'hydrogen': 1, 'gas': -1.43},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'capacityLim': 100e3, 'techUnitPower': 320
                                },
                               }
                         )
        )

        tech = "SMR + CCS2"
        capex, opex, LifeSpan = 1000e3, 50e3, 60
        scenario['conversionTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Hydrogen production',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                                'minCapacity': 0, 'maxCapacity': 100e3,
                                'EmissionCO2': -268, 'Conversion': {'electricity': -0.34, 'hydrogen': 1, 'gas': -1.43},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'capacityLim': 100e3, 'techUnitPower': 320
                                },
                               }
                         )
        )

        tech = "CCS1"
        capex, opex, LifeSpan = 100e3, 0e3, 60
        scenario['conversionTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Carbon capture',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex,
                                'operationCost': opex, 'capacityLim': 100e3, 'techUnitPower': 320},
                               }
                         )
        )

        tech = "CCS2"
        capex, opex, LifeSpan = 100e3, 0e3, 60
        scenario['conversionTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'Category': 'Carbon capture',
                                'LifeSpan': LifeSpan, 'powerCost': 0, 'investCost': capex,
                                'operationCost': opex, 'capacityLim': 100e3, 'techUnitPower': 320},
                               }
                         )
        )

scenario['conversionTechs'] = pd.concat(scenario['conversionTechs'], axis=1)

scenario['storageTechs'] = []
for k, year in enumerate(yearList):
    tech = "Battery"
    capex1, opex1, LifeSpan = tech_eco_data.get_capex_new_tech_RTE(
        tech + ' - 1h', hyp='ref', year=year)
    capex4, opex4, LifeSpan = tech_eco_data.get_capex_new_tech_RTE(
        tech + ' - 4h', hyp='ref', year=year)
    capex_per_kWh = (capex4 - capex1) / 3
    capex_per_kW = capex1 - capex_per_kWh
scenario['storageTechs'] = []
for area in areaList:
    for k, year in enumerate(yearList):
        tech = "Battery"
        capex1, opex1, LifeSpan = tech_eco_data.get_capex_new_tech_RTE(
            tech + ' - 1h', hyp='ref', year=year)
        capex4, opex4, LifeSpan = tech_eco_data.get_capex_new_tech_RTE(
            tech + ' - 4h', hyp='ref', year=year)
        capex_per_kWh = (capex4 - capex1) / 3
        capex_per_kW = capex1 - capex_per_kWh

        scenario['storageTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year, 'storageResource': 'electricity',  # ambiguïté du nom des paramètres ?
                                'storageLifeSpan': LifeSpan,
                                'storagePowerCost': capex_per_kW,
                                'storageEnergyCost': capex_per_kWh,
                                # TODO: according to RTE OPEX seems to vary with energy rather than power
                                'storageOperationCost': opex1,
                                'p_max': 5000,
                                'c_max': 50000,
                                'storageChargeFactors': {'electricity': 0.9200},
                                'storageDischargeFactors': {'electricity': 1.09},
                                'storageDissipation': 0.0085,
                                },
                               }
                         )
        )

        tech = "Salt cavern"
        scenario['storageTechs'].append(
            pd.DataFrame(data={tech:
                               {'AREA': area, 'YEAR': year,
                                'storageResource': 'hydrogen',
                                'storageLifeSpan': 40,
                                'storagePowerCost': 0,
                                'storageEnergyCost': 350e3,
                                'storageOperationCost': 2e3,
                                'p_max': 10000,
                                'c_max': stockage_h_area(area),
                                'storageChargeFactors': {'electricity': 0.0168, 'hydrogen': 1.0},
                                'storageDischargeFactors': {'hydrogen': 1.0},
                                'storageDissipation': 0,
                                },
                               }
                         )
        )


scenario['storageTechs'] = pd.concat(scenario['storageTechs'], axis=1)

scenario['transportTechs'] = []
for k, year in enumerate(yearList):
    ttech = 'Pipeline_S'
    p_max = 50000.
    capex, opex, LifeSpan = 1583, 3e-4, 40
    scenario['transportTechs'].append(
        pd.DataFrame(data={ttech:
                           {'YEAR': year, 'transportResource': 'hydrogen',
                            'transportLifeSpan': LifeSpan, 'transportPowerCost': 9e-5, 'transportInvestCost': capex, 'transportOperationCost': opex,
                            # 'transportMinPower':1., 'transportMaxPower': p_max,
                            'transportEmissionCO2': 0,
                            'transportChargeFactors': {'hydrogen': 5e-3},
                            'transportDischargeFactors': {'hydrogen': 5e-3},
                            'transportDissipation': 2e-5,
                            # puissance maximale de fonctionnement du pipeline (=débit max), fixée
                            'transportUnitPower': p_max_fonc[ttech]
                            }
                           }
                     )
    )

    ttech = 'Pipeline_M'
    p_max = 50000.
    capex, opex, LifeSpan = 638, 1.2e-4, 40
    scenario['transportTechs'].append(
        pd.DataFrame(data={ttech:
                           {'YEAR': year, 'transportResource': 'hydrogen',
                            'transportLifeSpan': LifeSpan, 'transportPowerCost': 3.2e-4, 'transportInvestCost': capex, 'transportOperationCost': opex,
                            # 'transportMinPower':1., 'transportMaxPower': p_max,
                            'transportEmissionCO2': 0,
                            'transportChargeFactors': {'hydrogen': 5e-3},
                            'transportDischargeFactors': {'hydrogen': 5e-3},
                            'transportDissipation': 2e-5,
                            # puissance maximale de fonctionnement du pipeline (=débit max), fixée
                            'transportUnitPower': p_max_fonc[ttech]
                            }
                           }
                     )
    )

    ttech = 'Pipeline_L'
    p_max = 50000.
    capex, opex, LifeSpan = 253, 3.4e-5, 40
    scenario['transportTechs'].append(
        pd.DataFrame(data={ttech:
                           {'YEAR': year, 'transportResource': 'hydrogen',
                            'transportLifeSpan': LifeSpan, 'transportPowerCost': 1.5e-3, 'transportInvestCost': capex, 'transportOperationCost': opex,
                            # 'transportMinPower':1., 'transportMaxPower': p_max,
                            'transportEmissionCO2': 0,
                            'transportChargeFactors': {'hydrogen': 5e-3},
                            'transportDischargeFactors': {'hydrogen': 5e-3},
                            'transportDissipation': 2e-5,
                            'transportUnitPower': p_max_fonc[ttech]
                            }
                           }
                     )
    )


# ttech = truck transporting hydrogen
for k, year in enumerate(yearList):
    ttech = 'truckTransportingHydrogen'
    p_max = 50000  # to change
    capex, opex, LifeSpan = 296, 7e-3, 10
    scenario['transportTechs'].append(
        pd.DataFrame(data={ttech:
                           {'YEAR': year, 'transportResource': 'hydrogen',
                            'transportLifeSpan': LifeSpan, 'transportPowerCost': 4.2e-2, 'transportInvestCost': capex, 'transportOperationCost': opex,
                            # 'transportMinPower':1, 'transportMaxPower': p_max,
                            'transportEmissionCO2': 1/23,
                            'transportChargeFactors': {'hydrogen': 0.1},
                            'transportDischargeFactors': {'hydrogen': 0.001},
                            'transportDissipation': 0.0,
                            'transportUnitPower': p_max_fonc[ttech]
                            }
                           }
                     )
    )


# ttech = truck transporting electricity
# ttech = electric cable


scenario['transportTechs'] = pd.concat(scenario['transportTechs'], axis=1)

scenario['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675, 0.165, nYears),
                                     index=yearList, columns=('carbonTax',))

scenario['carbonGoals'] = pd.DataFrame(data=np.linspace(974e6, 205e6, nYears),
                                       index=yearList, columns=('carbonGoals',))

scenario['maxBiogasCap'] = pd.DataFrame(data=np.linspace(0, 310e6, nYears),
                                        index=yearList, columns=('maxBiogasCap',))

scenario['gridConnection'] = pd.read_csv("Data/Raw/CalendrierHPHC_TIME.csv", sep=',', decimal='.', skiprows=0,
                                         comment="#").set_index(["TIMESTAMP"]).loc[t]

scenario['economicParameters'] = pd.DataFrame({
    'discountRate': [0.04],
    'financeRate': [0.04]
}
)


df_res_ref = pd.read_csv('./Data/Raw/set2020-2050_horaire_TIMExRESxYEAR.csv',
                         sep=',', decimal='.', skiprows=0, comment="#").set_index(["YEAR", "TIMESTAMP", 'RESOURCES'])
t8760 = df_res_ref.index.get_level_values('TIMESTAMP').unique().values

# en €/MWh
# 4.5€ le kg d'H soit les 33*e-3 MWh donc 4.5/(33*e-3)€ le MWh
prix_kg = 0.0000005
prix_MWh = prix_kg/ (33 * 1e-3)

scenario['resourceImportPrices'] = pd.concat(
    (
        pd.DataFrame(data={
            'AREA': area,
            'YEAR': year,
            'TIMESTAMP': t,
            'electricity': np.interp(t, t8760, df_res_ref.loc[(year, slice(None), 'electricity'), 'importCost'].values),
            'natural gas': 2 * np.interp(t, t8760, df_res_ref.loc[(year, slice(None), 'gazNat'), 'importCost'].values),
            'biogas': 150 * np.ones(nHours),

            'hydrogen': prix_MWh * np.ones(nHours), # à changer
        }) for k, year in enumerate(yearList[1:])
        for area in areaList
    )
)

scenario['resourceImportCO2eq'] = pd.concat(
    (
        pd.DataFrame(data={
            'AREA': area,
            'YEAR': year,
            'TIMESTAMP': t,
            'electricity': np.interp(t, t8760, df_res_ref.loc[(year, slice(None), 'electricity'), 'emission'].values),
            # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
            'gas': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1 + 203.5 * (1 - tech_eco_data.get_biogas_share_in_network_RTE(year)),
            # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
            'natural gas': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1 + 203.5 * (1 - tech_eco_data.get_biogas_share_in_network_RTE(year)),
            'biogas': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1,
            # Taking 100 yr GWP of H2 and 5% losses due to upstream leaks. Leaks fall to 2% in 2050 See: https://www.energypolicy.columbia.edu/research/commentary/hydrogen-leakage-potential-risk-hydrogen-economy
            'hydrogen': max(0, 0.05 - .03 * (year - yearZero)/(2050 - yearZero)) * 11 / 33,
        }) for k, year in enumerate(yearList[1:])
        for area in areaList
    )
)

scenario['convTechList'] = ["Offshore wind - floating", "Onshore wind",
                            "Ground PV", "ElectrolysisS", "ElectrolysisM", "ElectrolysisL"]
ctechs = scenario['convTechList']
availabilityFactor = pd.read_csv('Data/Raw/availabilityFactor2020-2050_PACA_TIMExTECHxYEAR - renamed.csv',
                                 sep=',', decimal='.', skiprows=0).set_index(["YEAR", "TIMESTAMP", "TECHNOLOGIES"]).loc[(slice(None), t, slice(None))]
itechs = availabilityFactor.index.isin(ctechs, level=2)
scenario['availability'] = availabilityFactor.loc[(
    slice(None), slice(None), itechs)]

# availability pour transport ?
ttechs_list = list(scenario['transportTechs'].columns.unique())

scenario["yearList"] = yearList
scenario["areaList"] = areaList
scenario["transitionFactors"] = pd.DataFrame(
    {'TECHNO1': ['Existing SMR', 'Existing SMR', 'SMR', 'SMR', 'SMR + CCS1'],
     'TECHNO2': ['SMR + CCS1', 'SMR + CCS2', 'SMR + CCS1', 'SMR + CCS2', 'SMR + CCS2'],
     'TransFactor': 1}).set_index(['TECHNO1', 'TECHNO2'])
