import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from scenarios import *

# CODE ANAELLE
# vlist = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDel_Pvar','capacityDem_Dvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
#          'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar', ]

# outputFolder = 'out'

# res = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in vlist}

# res['Annual production'] = res['power_Dvar'].set_index(['YEAR_op', 'TIMESTAMP','TECHNOLOGIES'])

# df = res['Annual production']

# yrs = df.index.get_level_values('YEAR_op').unique()
# techs = df.index.get_level_values('TECHNOLOGIES').unique()
# Pel = df.groupby(['YEAR_op','TECHNOLOGIES']).sum()
# print(Pel)
# tech = 'Electrolysis'
# plt.plot(yrs, Pel.loc[(yrs,tech),].values / 1e6,'s', label=tech)

# plt.xlabel('Year')
# plt.ylabel('Annual production (TWh)')
# plt.xticks(yrs)
# plt.legend()

# plt.show()

def print_global_H2_bar(power_Dvar='out_scenario1/power_Dvar.csv'):
    YEAR = [2030, 2040, 2050]
    df = pd.read_csv(power_Dvar)

    # la production de H2
    tech_H2 = ["ElectrolysisL", "ElectrolysisM",
               "ElectrolysisS", "SMR + CCS1", "SMR + CCS2", "SMR"]
    colors_dict = {"ElectrolysisL": 'darkred', "ElectrolysisM": 'firebrick',
                   "ElectrolysisS": 'lightcoral', "SMR + CCS1": "dimgray", "SMR + CCS2": 'darkgray', "SMR": 'silver'}
    by_techno = df.groupby(by='TECHNOLOGIES')
    global_dict = {}
    for techno in tech_H2:
        global_dict[techno] = []
        for y in YEAR:
            global_dict[techno].append(by_techno.get_group(
                techno)[by_techno.get_group(techno)['YEAR_op'] == y]['power_Dvar'].sum())
    # # Les subplots
    # fig, axis = plt.subplots(2,2)

    # On print la production de H2
    for techno in tech_H2:
        plt.bar(YEAR, global_dict[techno],
                color=colors_dict[techno], label=techno)
    plt.legend()
    plt.xticks(YEAR)
    plt.xlabel('Year')
    plt.ylabel('H2 production (MWh)')
    plt.title('Global H2 production')

    # # la production d'éléctricité
    # tech_elec= ["Ground PV", "Offshore wind - floating", "Onshore wind"]
    # colors_dict_elec={"Ground PV":"lime", "Offshore wind - floating":"mediumturquoise", "Onshore wind":"royalblue"}

    # for techno_elec in tech_elec:
    #     global_dict[techno_elec] = []
    #     for y in YEAR:
    #         global_dict[techno_elec].append(by_techno.get_group(techno_elec)[by_techno.get_group(techno_elec)['YEAR_op']==y]['power_Dvar'].sum())

    # # On print
    # for techno_elec in tech_elec:
    #     axis[0,1].bar(YEAR, global_dict[techno_elec], color = colors_dict_elec[techno_elec], label=techno_elec )
    # plt.legend()
    # plt.xticks(YEAR)
    # plt.title("Production d'éléctricité globale")

def print_global_elec_bar_instant(power_Dvar='out/power_Dvar.csv'):
    """A faire"""
    YEAR = [2030, 2040, 2050]
    df = pd.read_csv(power_Dvar)

    # la production de H2 
    tech_H2= ["ElectrolysisL", "ElectrolysisM", "ElectrolysisS", "SMR + CCS1", "SMR + CCS2", "SMR", "Existing SMR"]
    colors_dict={"ElectrolysisL":'darkred', "ElectrolysisM":'firebrick', "ElectrolysisS":'lightcoral', "SMR + CCS1":"dimgray", "SMR + CCS2" : 'darkgray', "SMR" : 'silver', "Existing SMR":"gray"  }
    by_techno = df.groupby(by='TECHNOLOGIES')
    global_dict = {}
    for techno in tech_H2:
        global_dict[techno] = []
        for y in YEAR:
            global_dict[techno].append(by_techno.get_group(
                techno)[by_techno.get_group(techno)['YEAR_op'] == y]['power_Dvar'].sum())
    # # Les subplots
    # fig, axis = plt.subplots(2,2)

    # On print la production de H2
    for techno in tech_H2:
        plt.bar(YEAR, global_dict[techno],
                color=colors_dict[techno], label=techno)
    plt.legend()
    plt.xticks(YEAR)
    plt.title('Production de H2 globale')

    # # la production d'éléctricité
    # tech_elec= ["Ground PV", "Offshore wind - floating", "Onshore wind"]
    # colors_dict_elec={"Ground PV":"lime", "Offshore wind - floating":"mediumturquoise", "Onshore wind":"royalblue"}

    # for techno_elec in tech_elec:
    #     global_dict[techno_elec] = []
    #     for y in YEAR:
    #         global_dict[techno_elec].append(by_techno.get_group(techno_elec)[by_techno.get_group(techno_elec)['YEAR_op']==y]['power_Dvar'].sum())

    # # On print
    # for techno_elec in tech_elec:
    #     axis[0,1].bar(YEAR, global_dict[techno_elec], color = colors_dict_elec[techno_elec], label=techno_elec )
    # plt.legend()
    # plt.xticks(YEAR)
    # plt.title("Production d'éléctricité globale")


def print_global_H2_camembert_instant(y, power_Dvar='out/power_Dvar.csv'):
    df = pd.read_csv(power_Dvar)

    # la production de H2
    tech_H2 = ["ElectrolysisL", "ElectrolysisM",
               "ElectrolysisS", "SMR + CCS1", "SMR + CCS2", "SMR"]
    colors_dict = {"ElectrolysisL": 'darkred', "ElectrolysisM": 'firebrick',
                   "ElectrolysisS": 'lightcoral', "SMR + CCS1": "dimgray", "SMR + CCS2": 'darkgray', "SMR": 'silver'}
    by_techno = df.groupby(by='TECHNOLOGIES')
    global_dict = {}
    for techno in tech_H2:
        global_dict[techno] = (by_techno.get_group(
            techno)[by_techno.get_group(techno)['YEAR_op'] == y]['power_Dvar'].sum())
    # on normalise
    total_power = 0
    for techno in tech_H2:
        total_power += global_dict[techno]
    if total_power == 0:
        return f"Il n'y a pas de production de H2 pour l'année {y}"
    for techno in global_dict:
        global_dict[techno] = global_dict[techno]/total_power
    # On print la production de H2

    plt.pie([global_dict[techno] for techno in tech_H2], colors=[
            colors_dict[techno] for techno in tech_H2], labels=tech_H2, labeldistance=None)
    plt.title(f"Part de la production de H2 sur l'année {y}")
    plt.legend()

def print_global_elec_camembert_instant(y, power_Dvar='out_scenario1/power_Dvar.csv'):
    df = pd.read_csv(power_Dvar)
    by_techno = df.groupby(by='TECHNOLOGIES')
    global_dict = {}
    # la production d'éléctricité
    tech_elec = ["Ground PV", "Offshore wind - floating", "Onshore wind"]
    colors_dict = {"Ground PV": "lime",
                   "Offshore wind - floating": "mediumturquoise", "Onshore wind": "royalblue"}

    for techno in tech_elec:
        global_dict[techno] = (by_techno.get_group(
            techno)[by_techno.get_group(techno)['YEAR_op'] == y]['power_Dvar'].sum())
    plt.pie([global_dict[techno] for techno in tech_elec], colors=[
            colors_dict[techno] for techno in tech_elec], labels=tech_elec, labeldistance=None)
    plt.title(f"Part de la production d'éléctricité sur l'année {y}")
    plt.legend()

def print_global_transport_camembert_instant(y, power_Dvar='out/transportFlowOut_Dvar.csv'):
    df = pd.read_csv(power_Dvar)
    by_techno = df.groupby(by='TRANS_TECHNO')
    global_dict = {}
        # la production d'éléctricité
    tech_transport= ["Pipeline_S", "Pipeline_M", "Pipeline_L", ]
    colors_dict={"Pipeline_S":"lime", "Pipeline_M":"mediumturquoise", "Pipeline_L":"royalblue"}

    try:
        for techno in tech_transport:
            global_dict[techno] = (by_techno.get_group(techno)[by_techno.get_group(techno)['YEAR_op']==y]['transportFlowOut_Dvar'].sum())
        plt.pie([global_dict[techno] for techno in   tech_transport], colors = [colors_dict[techno] for techno in   tech_transport], labels=  tech_transport, labeldistance = None )
        plt.title(f"Part du transport d'H2 sur l'année {y}")
        plt.legend()
    except :
        return f"Il n'y a pas de transport sur l'année {y}"
   
<<<<<<< HEAD
    
def prod_H2_an(y, file = "out/power_Dvar.csv", tot = False) :
=======

def print_global_H2_bar_install(power_Dvar='out/capacity_Pvar.csv'):
    YEAR = [2030, 2040, 2050]
    df = pd.read_csv(power_Dvar)

    # la production de H2 
    tech_H2= ["ElectrolysisL", "ElectrolysisM", "ElectrolysisS", "SMR + CCS1", "SMR + CCS2", "SMR"]
    colors_dict={"ElectrolysisL":'darkred', "ElectrolysisM":'firebrick', "ElectrolysisS":'lightcoral', "SMR + CCS1":"dimgray", "SMR + CCS2" : 'darkgray', "SMR" : 'silver'  }
    by_techno = df.groupby(by='TECHNOLOGIES')
    global_dict = {}
    for techno in tech_H2:
        global_dict[techno] = []
        for y in YEAR:
            global_dict[techno].append(by_techno.get_group(techno)[by_techno.get_group(techno)['YEAR_op']==y]['capacity_Pvar'].sum())

    # On print la production de H2
    for techno in tech_H2:
        plt.bar(YEAR, global_dict[techno], color = colors_dict[techno], label=techno )
    plt.legend()
    plt.xticks(YEAR)
    plt.xlabel('Year')
    plt.ylabel('H2 production (MW)')
    plt.title('Global H2 production')

def print_global_elec_bar_install(power_Dvar='out/capacity_Pvar.csv'):
    YEAR = [2030, 2040, 2050]
    df = pd.read_csv(power_Dvar)

    # la production d'elec 
    tech_elec= ["Ground PV", "Offshore wind - floating", "Onshore wind"]
    colors_dict={"Ground PV":"lime", "Offshore wind - floating":"mediumturquoise", "Onshore wind":"royalblue"}
    by_techno = df.groupby(by='TECHNOLOGIES')
    global_dict = {}
    for techno in tech_elec:
        global_dict[techno] = []
        for y in YEAR:
            global_dict[techno].append(by_techno.get_group(techno)[by_techno.get_group(techno)['YEAR_op']==y]['capacity_Pvar'].sum())

    # On print la production de H2
    for techno in tech_elec:
        plt.bar(YEAR, global_dict[techno], color = colors_dict[techno], label=techno )
    plt.legend()
    plt.xticks(YEAR)
    plt.xlabel('Year')
    plt.ylabel('electricity capacity production (MW)')
    plt.title('Global electricity capacity production')

def print_global_transport_bar_install(power_Dvar='out/transportFlowOut_Dvar.csv'):
    YEAR = [2030, 2040, 2050]
    df = pd.read_csv(power_Dvar)

    # la production d'elec 
    tech_elec= ["Ground PV", "Offshore wind - floating", "Onshore wind"]
    colors_dict={"Ground PV":"lime", "Offshore wind - floating":"mediumturquoise", "Onshore wind":"royalblue"}
    by_techno = df.groupby(by='TECHNOLOGIES')
    global_dict = {}
    for techno in tech_elec:
        global_dict[techno] = []
        for y in YEAR:
            global_dict[techno].append(by_techno.get_group(techno)[by_techno.get_group(techno)['YEAR_op']==y]['capacity_Pvar'].sum())

    # On print la production de H2
    for techno in tech_elec:
        plt.bar(YEAR, global_dict[techno], color = colors_dict[techno], label=techno )
    plt.legend()
    plt.xticks(YEAR)
    plt.xlabel('Year')
    plt.ylabel('electricity capacity production (MW)')
    plt.title('Global electricity capacity production')


def print_global_elec_bar_install_capex(power_Dvar='out/capacityCosts_Pvar.csv'):
    YEAR = [2030, 2040, 2050]
    df = pd.read_csv(power_Dvar)

    # la production d'elec 
    tech_elec= ["Ground PV", "Offshore wind - floating", "Onshore wind"]
    colors_dict={"Ground PV":"lime", "Offshore wind - floating":"mediumturquoise", "Onshore wind":"royalblue"}
    by_techno = df.groupby(by='TECHNOLOGIES')
    global_dict = {}
    for techno in tech_elec:
        global_dict[techno] = []
        for y in YEAR:
            global_dict[techno].append(by_techno.get_group(techno)[by_techno.get_group(techno)['YEAR_op']==y]['capacityCosts_Pvar'].sum())

    # On print la production de H2
    for techno in tech_elec:
        plt.bar(YEAR, global_dict[techno], color = colors_dict[techno], label=techno )
    plt.legend()
    plt.xticks(YEAR)
    plt.xlabel('Year')
    plt.ylabel('electricity price (€)')
    plt.title('Global electricity price')
   
def print_global_H2_bar_install_capex(power_Dvar='out/capacityCosts_Pvar.csv'):
    YEAR = [2030, 2040, 2050]
    df = pd.read_csv(power_Dvar)

    # la production de H2 
    tech_H2= ["ElectrolysisL", "ElectrolysisM", "ElectrolysisS", "SMR + CCS1", "SMR + CCS2", "SMR"]
    colors_dict={"ElectrolysisL":'darkred', "ElectrolysisM":'firebrick', "ElectrolysisS":'lightcoral', "SMR + CCS1":"dimgray", "SMR + CCS2" : 'darkgray', "SMR" : 'silver'  }
    by_techno = df.groupby(by='TECHNOLOGIES')
    global_dict = {}
    for techno in tech_H2:
        global_dict[techno] = []
        for y in YEAR:
            global_dict[techno].append(by_techno.get_group(techno)[by_techno.get_group(techno)['YEAR_op']==y]["capacityCosts_Pvar"].sum())

    # On print la production de H2
    for techno in tech_H2:
        plt.bar(YEAR, global_dict[techno], color = colors_dict[techno], label=techno )
    plt.legend()
    plt.xticks(YEAR)
    plt.xlabel('Year')

    plt.ylabel('H2 cost (€)')
    plt.title('Global H2 Cost')
  
def prod_H2_an(file = "out_scenario1/power_Dvar.csv") :
>>>>>>> df3e48396deec17842061beff77a35c3a30c1fdd
    tech_H2= ["ElectrolysisL", "ElectrolysisM", "ElectrolysisS", "SMR + CCS1", "SMR + CCS2", "SMR"]
    colors_dict={"ElectrolysisL":'darkred', "ElectrolysisM":'firebrick', "ElectrolysisS":'lightcoral', "SMR + CCS1":"dimgray", "SMR + CCS2" : 'darkgray', "SMR" : 'silver'  }
    
    df = pd.read_csv(file)
    if not tot :
        df = df[df['YEAR_op'] == y]
    by_techno = df.pivot_table(index='AREA',
    columns='TECHNOLOGIES',
    values='power_Dvar',
    aggfunc = 'sum')
    print(by_techno)
    plt.figure()
    by_techno[tech_H2].reset_index().plot(
        x = 'AREA',
        kind = 'bar',
        stacked=True,
        title="Puissance instantanée par moyen de transformation par année"
    )
 
<<<<<<< HEAD
def energy_H2(file = "out_scenario1bis/energy_Pvar.csv") :
    df = pd.read_csv(file)
    print(df.pivot_table(
        index = 'RESOURCES',
        columns = 'YEAR_op',
        values = 'energy_Pvar',
        aggfunc = 'sum'
    ))


prod_H2_an(2050, tot=False)
=======

print_global_H2_bar_install_capex()
>>>>>>> df3e48396deec17842061beff77a35c3a30c1fdd
plt.show()


