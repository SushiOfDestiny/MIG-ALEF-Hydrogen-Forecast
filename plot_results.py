import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 


from scenarios import scenario

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

def print_global_H2_bar(power_Dvar='out/power_Dvar.csv'):
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
            global_dict[techno].append(by_techno.get_group(techno)[by_techno.get_group(techno)['YEAR_op']==y]['power_Dvar'].sum())
    # # Les subplots 
    # fig, axis = plt.subplots(2,2)

    # On print la production de H2
    for techno in tech_H2:
        plt.bar(YEAR, global_dict[techno], color = colors_dict[techno], label=techno )
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
        
        



def print_global_elec_hugo(power_Dvar='out/power_Dvar.csv'):
    """A faire"""
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
            global_dict[techno].append(by_techno.get_group(techno)[by_techno.get_group(techno)['YEAR_op']==y]['power_Dvar'].sum())
    # # Les subplots 
    # fig, axis = plt.subplots(2,2)

    # On print la production de H2
    for techno in tech_H2:
        plt.bar(YEAR, global_dict[techno], color = colors_dict[techno], label=techno )
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




def print_global_H2_camembert(y, power_Dvar='out/power_Dvar.csv'):
    df = pd.read_csv(power_Dvar)

    # la production de H2 
    tech_H2= ["ElectrolysisL", "ElectrolysisM", "ElectrolysisS", "SMR + CCS1", "SMR + CCS2", "SMR"]
    colors_dict={"ElectrolysisL":'darkred', "ElectrolysisM":'firebrick', "ElectrolysisS":'lightcoral', "SMR + CCS1":"dimgray", "SMR + CCS2" : 'darkgray', "SMR" : 'silver'  }
    by_techno = df.groupby(by='TECHNOLOGIES')
    global_dict = {}
    for techno in tech_H2:  
        global_dict[techno]=(by_techno.get_group(techno)[by_techno.get_group(techno)['YEAR_op']==y]['power_Dvar'].sum())
    # on normalise
    total_power = 0
    for techno in tech_H2:
        total_power += global_dict[techno]
    if total_power ==0:
        return f"Il n'y a pas de production de H2 pour l'année {y}"
    for techno in global_dict:
        global_dict[techno] = global_dict[techno]/total_power
    # On print la production de H2
    
    plt.pie([global_dict[techno] for techno in tech_H2], colors = [colors_dict[techno] for techno in tech_H2], labels=tech_H2, labeldistance = None )
    plt.title(f"Part de la production de H2 sur l'année {y}")
    plt.legend()
    








print_global_H2_camembert(2030)

plt.show()

