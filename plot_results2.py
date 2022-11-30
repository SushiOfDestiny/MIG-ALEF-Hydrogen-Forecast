import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from scenarios import *

# façon la plus simple pour afficher une dataframe


vlist = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar', 'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar',
         'Tmaxtot_Pvar']




def show_power_Dvar(outputFolder = 'out_scenario1'):
    '''shows the 'power_Dvar' by year and area from all tech'''

    res = {v: pd.read_csv(outputFolder + '/' + v +
                      '.csv').drop(columns='Unnamed: 0') for v in vlist}

    df = res['power_Dvar'].set_index(
        ['YEAR_op', 'TIMESTAMP', 'TECHNOLOGIES', 'AREA'])
    df.dropna(inplace=True)

    df_year = pd.DataFrame()
    df_year_area = df.groupby(['YEAR_op', 'AREA']).sum()

    dic = {}
    for ville in areaList:
            dic[ville] = df_year_area.loc[(slice(None), ville), :]
            df_year[ville] = pd.DataFrame(dic[ville].values, index=dic[ville].index.droplevel(
                1), columns=['power_Dvar'])['power_Dvar']

    df_year /= 1e3
    df_year.plot(kind='bar')
    plt.ylabel('GWh')
    plt.legend()
    plt.title(f'puissance de fonctionnement des technologies avec {outputFolder}')
    plt.savefig(f'show_power_Dvar_{outputFolder}')


def show_import_Dvar(outputFolder = 'out_scenario1'):
    """shows import by year by ressource by area"""
    res = {v: pd.read_csv(outputFolder + '/' + v +
                      '.csv').drop(columns='Unnamed: 0') for v in vlist}

    df = res['importation_Dvar'].set_index(
        ['YEAR_op', 'TIMESTAMP', 'RESOURCES', 'AREA'])
    df.dropna(inplace=True)

    df2 = df.groupby(['YEAR_op', 'RESOURCES', 'AREA']).sum()

    # for electricity and hydrogen by year and area
    for variable in ['electricity', 'hydrogen']:
        df3 = df2.loc[(slice(None), variable, slice(None)), :]
        df4 = pd.DataFrame(df3.values,
                          index=df3.index.droplevel(1),
                          columns=['importation_Dvar']
                          )

        df_year = pd.DataFrame()
        dic = {}
        for ville in areaList:
            dic[ville] = df4.loc[(slice(None), ville), :]
            df_year[ville] = pd.DataFrame(dic[ville].values, index=dic[ville].index.droplevel(
                1), columns=['importation_Dvar'])['importation_Dvar']

        # affichage
        # return df_year
        df_year /= 1e3
        df_year.plot(kind='bar')
        plt.ylabel('GWh')
        plt.legend()
        plt.title(f'importation annuelle de {variable} avec {outputFolder}')
        plt.savefig(f'show_import_Dvar_{variable}_{outputFolder}')


def show_storageConsumption_Pvar(outputFolder = 'out_scenario1'):
    """shows storageConsumption_Pvar by year by ressource by area for all stech"""
    res = {v: pd.read_csv(outputFolder + '/' + v +
                      '.csv').drop(columns='Unnamed: 0') for v in vlist}
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

    # affichage
    df_year /= 1e3
    df_year.plot(kind='bar')
    plt.ylabel('GWh')
    plt.legend()
    plt.title(f'storageConsumption_Pvar_{outputFolder}')
    plt.savefig(f'storageConsumption_Pvar_{outputFolder}')


# def show_Tmax_tot():

#     unitPower=scenario['transportTechs'].loc['transportUnitPower',:]
#     df = res['Tmaxtot_Pvar']

#     df2=pd.pivot_table(
#         data=df,
#         index=['YEAR_invest','AREA','AREA.1'],
#         columns=['TRANS_TECHNO','TmaxTot_Pvar'])
#     # data=res['Tmaxtot_Pvar'][res['Tmaxtot_Pvar']['YEAR_invest']

#     for y in yearList:
#         for a,a1 in couples_noeuds:



for i in range(1,3):
    show_power_Dvar(f'out_scenario{i}')
    show_import_Dvar(f'out_scenario{i}')
plt.show()