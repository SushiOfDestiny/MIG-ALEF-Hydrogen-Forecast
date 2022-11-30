import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from scenarios import *

vlist = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar', 'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar']

outputFolder = 'out'

res = {v: pd.read_csv(outputFolder + '/' + v +
                      '.csv').drop(columns='Unnamed: 0') for v in vlist}


# shows the 'power_Dvar' by year and area from all tech
df = res['power_Dvar'].set_index(
    ['YEAR_op', 'TIMESTAMP', 'TECHNOLOGIES', 'AREA'])
df.dropna(inplace=True)

df_year = df.groupby(['YEAR_op']).sum()
df_year_area = df.groupby(['YEAR_op', 'AREA']).sum()

df_year_fos = df_year_area.loc[(slice(None), 'Fos'), :]
df_year['power_Dvar_fos'] = pd.DataFrame(
    df_year_fos.values, index=df_year_fos.index.droplevel(1), columns=['power_Dvar'])['power_Dvar']
df_year_nice = df_year_area.loc[(slice(None), 'Nice'), :]
df_year['power_Dvar_nice'] = pd.DataFrame(
    df_year_nice.values, index=df_year_nice.index.droplevel(1), columns=['power_Dvar'])['power_Dvar']

df_year.plot(kind='bar')
plt.title('power_Dvar')

# shows storageConsumption_Pvar by year by ressource for all stech
df = res['storageConsumption_Pvar'].set_index(
    ['YEAR_op', 'TIMESTAMP', 'STOCK_TECHNO', 'AREA'])
df.dropna(inplace=True)
df_year = df.groupby(['YEAR_op']).sum()
df_year_area = df.groupby(['YEAR_op', 'AREA']).sum()

df_year_fos = df_year_area.loc[(slice(None), 'Fos'), :]
df_year['storageConsumption_Pvar_fos'] = pd.DataFrame(df_year_fos.values, index=df_year_fos.index.droplevel(
    1), columns=['storageConsumption_Pvar'])['storageConsumption_Pvar']
df_year_nice = df_year_area.loc[(slice(None), 'Nice'), :]
df_year['storageConsumption_Pvar_nice'] = pd.DataFrame(df_year_nice.values, index=df_year_nice.index.droplevel(
    1), columns=['storageConsumption_Pvar'])['storageConsumption_Pvar']

df_year.plot(kind='bar')
plt.title('storageConsumption_Pvar')

# shows import
df = res['importation_Dvar'].set_index(
    ['YEAR_op', 'TIMESTAMP', 'RESOURCES', 'AREA'])
df.dropna(inplace=True)
df_year_res_area = df.groupby(['YEAR_op', 'RESOURCES', 'AREA']).sum()

# for electricity and hydrogen by year and area
for variable in ['electricity', 'hydrogen']:
    dff = df_year_res_area.loc[(slice(None), variable, slice(None)), :]
    df = pd.DataFrame(dff.values,
                      index=dff.index.droplevel(1),
                      columns=['importation_Dvar']
                      )

    df_year_fos = df.loc[(slice(None), 'Fos'), :]
    df_year = pd.DataFrame()
    df_year['importation_Dvar_fos'] = pd.DataFrame(df_year_fos.values, index=df_year_fos.index.droplevel(
        1), columns=['importation_Dvar'])['importation_Dvar']
    df_year_nice = df.loc[(slice(None), 'Nice'), :]
    df_year['importation_Dvar_nice'] = pd.DataFrame(df_year_nice.values, index=df_year_nice.index.droplevel(
        1), columns=['importation_Dvar'])['importation_Dvar']

    df_year.plot(kind='bar')
    plt.title(f'importation_Dvar {variable}')


# AFFICHAGE
# def affichage():
# plt.xlabel('Year')
# plt.ylabel('Annual production (TWh)')
# plt.xticks(yrs)
# plt.legend()

plt.show()
