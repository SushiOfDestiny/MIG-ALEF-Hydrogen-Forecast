import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 


from scenarios import scenario

vlist = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDel_Pvar','capacityDem_Dvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']

outputFolder = 'out'

res = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in vlist}

res['Annual production'] = res['power_Dvar'].set_index(['YEAR_op', 'TIMESTAMP','TECHNOLOGIES'])

df = res['Annual production']

yrs = df.index.get_level_values('YEAR_op').unique()
techs = df.index.get_level_values('TECHNOLOGIES').unique()
Pel = df.groupby(['YEAR_op','TECHNOLOGIES']).sum()
print(Pel)
tech = 'Electrolysis'
plt.plot(yrs, Pel.loc[(yrs,tech),].values / 1e6,'s', label=tech)

plt.xlabel('Year')
plt.ylabel('Annual production (TWh)')
plt.xticks(yrs)
plt.legend()

plt.show() 
