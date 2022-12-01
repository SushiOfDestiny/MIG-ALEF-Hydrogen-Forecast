import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from scenarios import *

# façon la plus simple pour afficher une dataframe


vlist = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar', 'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar',
         'Tmaxtot_Pvar']

def create_res_dict(outputFolder):
    """returns result dictionary where each key is a csv file's name and the correspondant value is the file"""
    return {v: pd.read_csv(outputFolder + '/' + v +
                          '.csv').drop(columns='Unnamed: 0') for v in vlist}

def split_df(variable, dff, entity_list, df_res):
    """add columns to df_res corresponding to sliced dataframes from dff depending on entity_list """
    dic = {}
    for entity in entity_list:
        dic[entity] = dff.loc[(slice(None), entity), :]
        df_res[entity] = pd.DataFrame(dic[entity].values, index=dic[entity].index.droplevel(
            1), columns=[variable])[variable]


def show_power_Dvar(outputFolder='out_scenario1'):
    '''shows the annual energy consumption by year and area from all tech'''

    res = create_res_dict(outputFolder)

    df = res['power_Dvar'].set_index(
        ['YEAR_op', 'TIMESTAMP', 'TECHNOLOGIES', 'AREA'])
    df.dropna(inplace=True)

    df_year = pd.DataFrame()
    # multiplication by timeStep because variable power_Dvar has an hourly resolution
    df_year_area = df.groupby(['YEAR_op', 'AREA']).sum() * timeStep 

    # dic = {}
    # for ville in areaList:
    #     dic[ville] = df_year_area.loc[(slice(None), ville), :]
    #     df_year[ville] = pd.DataFrame(dic[ville].values, index=dic[ville].index.droplevel(
    #         1), columns=['power_Dvar'])['power_Dvar']
    
    split_df('power_Dvar',df_year_area,areaList,df_year)

    df_year /= 1e6
    df_year.plot(kind='bar')
    plt.ylabel('TWh')
    plt.legend()
    plt.title(
        f'énergie annuelle de fonctionnement des technologies avec scénario {outputFolder[-1]}')
    plt.savefig(f'show_power_Dvar_{outputFolder}')


def show_import_Dvar(outputFolder='out_scenario1'):
    """shows import by year by ressource by area"""
    res = create_res_dict(outputFolder)

    df = res['importation_Dvar'].set_index(
        ['YEAR_op', 'TIMESTAMP', 'RESOURCES', 'AREA'])
    df.dropna(inplace=True)

    df2 = df.groupby(['YEAR_op', 'RESOURCES', 'AREA']).sum() * timeStep

    # for electricity and hydrogen by year and area
    for resource in ['electricity', 'hydrogen']:
        df3 = df2.loc[(slice(None), resource, slice(None)), :]
        df4 = pd.DataFrame(df3.values,
                           index=df3.index.droplevel(1),
                           columns=['importation_Dvar']
                           )

        df_year = pd.DataFrame()
        # dic = {}
        # for ville in areaList:
        #     dic[ville] = df4.loc[(slice(None), ville), :]
        #     df_year[ville] = pd.DataFrame(dic[ville].values, index=dic[ville].index.droplevel(
        #         1), columns=['importation_Dvar'])['importation_Dvar']

        split_df('import_Dvar',df4,areaList,df_year)

        # affichage
        # return df_year
        df_year /= 1e6
        df_year.plot(kind='bar')
        plt.ylabel('TWh')
        plt.legend()
        # juste pour présentation
        plt.title(
            f'importation annuelle de {resource} avec scénario {outputFolder[-1]}')
        plt.savefig(f'show_import_Dvar_{resource}_{outputFolder}')


def show_storageConsumption_Pvar(outputFolder='out_scenario1'):
    """shows storageConsumption_Pvar by year by ressource by area for all stech,
    to be upgraded"""
    res = create_res_dict(outputFolder)

    df = res['storageConsumption_Pvar'].set_index(
        ['YEAR_op', 'TIMESTAMP', 'STOCK_TECHNO', 'AREA'])
    df.dropna(inplace=True)
    df_year = df.groupby(['YEAR_op']).sum() * timeStep
    df_year_area = df.groupby(['YEAR_op', 'AREA']).sum() * timeStep

    df_year_fos = df_year_area.loc[(slice(None), 'Fos'), :]
    df_year['storageConsumption_Pvar_fos'] = pd.DataFrame(df_year_fos.values, index=df_year_fos.index.droplevel(
        1), columns=['storageConsumption_Pvar'])['storageConsumption_Pvar']
    df_year_nice = df_year_area.loc[(slice(None), 'Nice'), :]
    df_year['storageConsumption_Pvar_nice'] = pd.DataFrame(df_year_nice.values, index=df_year_nice.index.droplevel(
        1), columns=['storageConsumption_Pvar'])['storageConsumption_Pvar']

    # affichage
    df_year /= 1e6
    df_year.plot(kind='bar')
    plt.ylabel('TWh')
    plt.legend()
    plt.title(
        f'consommation annuelle des installations de stockage avec scénario {outputFolder[-1]}')
    plt.savefig(f'storageConsumption_Pvar_{outputFolder}')


def show_Tmax_tot(outputFolder='out_scenario1'):
    """show transport units by year and ttech from all transport axes"""
    res = create_res_dict(outputFolder)

    df = res['Tmaxtot_Pvar'].set_index(
        ['YEAR_invest', 'TRANS_TECHNO', 'AREA', 'AREA.1'])
    df.dropna(inplace=True)

    df_year = df.groupby(['YEAR_invest']).sum()
    df_year.columns = ['Total']
    df2 = df.groupby(['YEAR_invest', 'TRANS_TECHNO']).sum()

    # dic = {}
    # for ttech in ttechs_list:
    #     dic[ttech] = df2.loc[(slice(None), ttech), :]
    #     df_year[ttech] = pd.DataFrame(dic[ttech].values, index=dic[ttech].index.droplevel(
    #         1), columns=['Tmaxtot_Pvar'])['Tmaxtot_Pvar']

    split_df('Tmaxtot_Pvar',df2,ttechs_list,df_year)

    df_year.columns = ["Total", "Pipeline_S", "Pipeline_M",
                       "Pipeline_L", "Camion transporteur d'hydrogène"]
    # affichage
    # return df_year

    # un axe (area1,area2) est aussi compté (area2,area1)
    df_year /= 2
    df_year.plot(kind='bar')
    plt.ylabel('nombre')
    plt.legend()
    plt.title(
        f'nombre annuel de moyens de transport avec scénario {outputFolder[-1]}')
    plt.savefig(f'show_Tmax_tot_Pvar_{outputFolder}')


def show_capacityCosts(outputFolder='out_scenario1'):
    """shows capacityCosts_Pvar (capex and opex) by year by ressource by area for all stech"""
    res = create_res_dict(outputFolder)

    df = res['capacityCosts_Pvar'].set_index(
        ['YEAR_op', 'TECHNOLOGIES', 'AREA'])
    df.dropna(inplace=True)

    # .rename({'capacityCosts_Pvar':'total'})
    df_year = df.groupby(['YEAR_op']).sum() * timeStep
    df_year.columns = ['Total']
    df2 = df.groupby(['YEAR_op', 'AREA']).sum() * timeStep

    dic = {}
    for ville in areaList:
        dic[ville] = df2.loc[(slice(None), ville), :]
        df_year[ville] = pd.DataFrame(dic[ville].values, index=dic[ville].index.droplevel(
            1), columns=['capacityCosts_Pvar'])['capacityCosts_Pvar']

    # affichage
    # return df_year
    df_year /= 1e9
    df_year.plot(kind='bar')
    plt.ylabel('Mrds €')
    plt.legend()
    plt.title(
        f'coût annuel des installations avec scénario {outputFolder[-1]}')
    plt.savefig(f'show_capacityCosts_Pvar_{outputFolder}')


# TRACE GRAPHES
for i in range(1, 3):
    show_power_Dvar(f'out_scenario{i}')
    show_import_Dvar(f'out_scenario{i}')
    show_capacityCosts(f'out_scenario{i}')
    show_Tmax_tot(f'out_scenario{i}')
plt.show()
