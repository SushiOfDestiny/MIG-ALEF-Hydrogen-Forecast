import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from scenarios import *

vlist = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar', 'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar']

outputFolder = 'out_scenario1'

res = {v: pd.read_csv(outputFolder + '/' + v +
                      '.csv').drop(columns='Unnamed: 0') for v in vlist}


def show_power_Dvar():
    '''shows the 'power_Dvar' by year and area from all tech'''
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

    df_year.plot(kind='bar')
    plt.ylabel('MWh')
    plt.legend()
    plt.title('puissance de fonctionnement des technologies')


def show_import_Dvar():
    """shows import by year by ressource by area"""
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

        df5 = pd.DataFrame()
        dic = {}
        for ville in areaList:
            dic[ville] = df4.loc[(slice(None), ville), :]
            df5[ville] = pd.DataFrame(dic[ville].values, index=dic[ville].index.droplevel(
                1), columns=['importation_Dvar'])['importation_Dvar']

        # affichage
        df5.plot(kind='bar')
        plt.ylabel('MWh')
        plt.legend()
        plt.title(f'importation annuelle de {variable}')


def show_storageConsumption_Pvar():
    """shows storageConsumption_Pvar by year by ressource by area for all stech"""
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
    df_year.plot(kind='bar')
    plt.ylabel('MWh')
    plt.legend()
    df_year.plot(kind='bar')
    plt.title('storageConsumption_Pvar')

show_power_Dvar()
show_import_Dvar()
plt.show()