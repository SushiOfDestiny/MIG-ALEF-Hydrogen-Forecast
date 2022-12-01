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
    plt.title(f'puissance de fonctionnement des technologies avec le scénario {i}')
    plt.savefig(f'show_power_Dvar_le scénario {i}')


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
        plt.title(f'importation annuelle de {variable} avec le scénario {i}')
        plt.savefig(f'show_import_Dvar_{variable}_le scénario {i}')


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
    plt.title(f'storageConsumption_Pvar_le scénario {i}')
    plt.savefig(f'storageConsumption_Pvar_le scénario {i}')


def show_Tmax_tot(outputFolder = 'out_scenario1'):
    """show max transport flow by year and ttech from all transport axes"""
    res = {v: pd.read_csv(outputFolder + '/' + v +
                      '.csv').drop(columns='Unnamed: 0') for v in vlist}
    df = res['Tmaxtot_Pvar'].set_index(['YEAR_invest', 'TRANS_TECHNO', 'AREA','AREA.1'])
    df.dropna(inplace=True)

    df_year = df.groupby(['YEAR_invest']).sum()
    df2 = df.groupby(['YEAR_invest', 'TRANS_TECHNO']).sum()

    dic = {}
    for ttech in ttechs_list:
        dic[ttech] = df2.loc[(slice(None), ttech), :]
        df_year[ttech] = pd.DataFrame(dic[ttech].values, index=dic[ttech].index.droplevel(
            1), columns=['capacityCosts_Pvar'])['capacityCosts_Pvar']

    # affichage
    # return df_year
    # df_year /= 1e3
    df_year.plot(kind='bar')
    plt.ylabel('MW')
    plt.legend()
    plt.title(f'puissance maximale annuelle des transports avec le scénario {i}')
    plt.savefig(f'show_Tmax_tot_Pvar_le scénario {i}')



def show_capacityCosts(outputFolder = 'out_scenario1'):
    """shows capacityCosts_Pvar (capex and opex) by year by ressource by area for all stech"""
    res = {v: pd.read_csv(outputFolder + '/' + v +
                      '.csv').drop(columns='Unnamed: 0') for v in vlist}
    df = res['capacityCosts_Pvar'].set_index(['YEAR_op', 'TECHNOLOGIES', 'AREA'])
    df.dropna(inplace=True)

    df_year = df.groupby(['YEAR_op']).sum() #.rename({'capacityCosts_Pvar':'total'})
    df2 = df.groupby(['YEAR_op', 'AREA']).sum()    

    
    dic = {}
    for ville in areaList:
        dic[ville] = df2.loc[(slice(None), ville), :]
        df_year[ville] = pd.DataFrame(dic[ville].values, index=dic[ville].index.droplevel(
            1), columns=['capacityCosts_Pvar'])['capacityCosts_Pvar']

    # affichage
    # return df_year
    df_year /= 1e3
    df_year.plot(kind='bar')
    plt.ylabel('GWh')
    plt.legend()
    plt.title(f'coût annuel des installations avec le scénario {i}')
    plt.savefig(f'show_capacityCosts_Pvar_le scénario {i}')

# TRACE GRAPHES
for i in range(1,3):
    # show_power_Dvar(f'out_scenario{i}')
    # show_import_Dvar(f'out_scenario{i}')
    # show_capacityCosts(f'out_scenario{i}')
    show_Tmax_tot(f'out_scenario{i}')
plt.show()