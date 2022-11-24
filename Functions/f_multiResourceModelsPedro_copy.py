from pyomo.environ import *
from pyomo.core import *
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def loadScenario(scenario, printTables=False):
    yearZero = scenario["yearList"][0]
    dy = scenario["yearList"][1] - yearZero
    # areaList = ...
    # sép par zone
    areaConsumption = scenario['resourceDemand'].melt(id_vars=['TIMESTAMP', 'YEAR', 'AREA'], var_name=[
                                                      'RESOURCES'], value_name='areaConsumption').set_index(['YEAR', 'TIMESTAMP', 'RESOURCES', 'AREA'])

    # para globaux début
    TechParameters = scenario['conversionTechs'].transpose().fillna(0)
    TechParameters.index.name = 'TECHNOLOGIES'
    TechParametersList = ['powerCost', 'operationCost', 'investCost', 'EnergyNbhourCap', 'minCapacity', 'maxCapacity',
                          'RampConstraintPlus', 'RampConstraintMoins', 'RampConstraintPlus2', 'RampConstraintMoins2', 'EmissionCO2', 'capacityLim']
    for k in TechParametersList:
        if k not in TechParameters:
            TechParameters[k] = 0
    TechParameters.drop(columns=['Conversion', 'Category'], inplace=True)
    TechParameters['yearStart'] = TechParameters['YEAR'] - \
        TechParameters['lifeSpan']//dy * dy
    TechParameters.loc[TechParameters['yearStart'] < yearZero, 'yearStart'] = 0
    TechParameters.set_index(['YEAR', TechParameters.index], inplace=True)

    StorageParameters = scenario['storageTechs'].transpose().fillna(0)
    StorageParameters.index.name = 'STOCK_TECHNO'
    StorageParametersList = [
        'resource', 'storagePowerCost', 'storageEnergyCost', 'p_max', 'c_max']
    for k in StorageParametersList:
        if k not in StorageParameters:
            StorageParameters[k] = 0
    StorageParameters.drop(
        columns=['chargeFactors', 'dischargeFactors', 'dissipation'], inplace=True)
    StorageParameters['storageYearStart'] = StorageParameters['YEAR'] - \
        round(StorageParameters['storagelifeSpan'] / dy) * dy
    StorageParameters.loc[StorageParameters['storageYearStart']
                          < yearZero, 'storageYearStart'] = 0
    StorageParameters.set_index(
        ['YEAR', StorageParameters.index], inplace=True)

    # ajout transport
    TransportParameters = scenario['transportTechs'].transpose().fillna(0)
    TransportParameters.index.name = 'TRANS_TECHNO'
    TransportParametersList = ['transportPowerCost', 'transportOperationCost', 'transportInvestCost', 'transportMinPower', 'transportMaxPower', 'transportEmissonCO2']
    for k in TransportParametersList:
        if k not in TransportParameters:
            TransportParameters[k] = 0
    TransportParameters.drop(columns=['transportChargeFactors', 'transportDischargeFactors','transportDissipation'], inplace=True)
    TransportParameters['yearStart'] = TransportParameters['YEAR'] - \
        TransportParameters['transportLifeSpan']//dy * dy
    TransportParameters.loc[TransportParameters['yearStart'] < yearZero, 'yearStart'] = 0
    TransportParameters.set_index(['YEAR', TransportParameters.index], inplace=True)

    CarbonTax = scenario['carbonTax'].copy()
    CarbonTax.index.name = 'YEAR'

    df_conv = scenario['conversionTechs'].transpose(
    ).set_index('YEAR', append=True)['Conversion']
    conversionFactor = pd.DataFrame(data={tech: df_conv.loc[(tech, 2020)] for tech in scenario['conversionTechs'].columns}).fillna(
        0)  # TODO: Take into account evolving conversion factors (for electrolysis improvement, for instance)
    conversionFactor.index.name = 'RESOURCES'
    conversionFactor = conversionFactor.reset_index('RESOURCES').melt(
        id_vars=['RESOURCES'], var_name='TECHNOLOGIES', value_name='conversionFactor').set_index(['RESOURCES', 'TECHNOLOGIES'])

    df_sconv = scenario['storageTechs'].transpose(
    ).set_index('YEAR', append=True)
    stechSet = set([k[0] for k in df_sconv.index.values])

    # c quoi df ?
    df = {}
    for k1, k2 in (('charge', 'In'),  ('discharge', 'Out')):
        # TODO: Take into account evolving conversion factors
        df[k1] = pd.DataFrame(data={tech: df_sconv.loc[(
            tech, 2020), k1 + 'Factors'] for tech in stechSet}).fillna(0)
        df[k1].index.name = 'RESOURCES'
        df[k1] = df[k1].reset_index(['RESOURCES']).melt(
            id_vars=['RESOURCES'], var_name='TECHNOLOGIES', value_name='storageFactor' + k2)

    df['dissipation'] = pd.concat(pd.DataFrame(
        data={'dissipation': [df_sconv.loc[(tech, 2020), 'dissipation']],
              'RESOURCES': df_sconv.loc[(tech, 2020), 'resource'],
              'TECHNOLOGIES': tech}) for tech in stechSet
    )
    storageFactors = pd.merge(
        df['charge'], df['discharge'], how='outer').fillna(0)
    storageFactors = pd.merge(storageFactors, df['dissipation'], how='outer').fillna(
        0).set_index(['RESOURCES', 'TECHNOLOGIES'])

    df_stransport = scenario['transportTechs'].transpose(
    ).set_index('YEAR', append=True)
    stranstechSet = set([k[0] for k in df_stransport.index.values])

    df2 = {}
    for k1, k2 in (('charge', 'In'), ('discharge', 'Out')):
        df2[k1] = pd.DataFrame(data={trans: df_stransport.loc[(
            trans, 2020), k1+'Factors'] for trans in stranstechSet}).fillna(0)
        df2[k1].index.name = 'RESOURCES'
        df2[k1] = df[k1].reset_index(['RESOURCES']).melt(
            id_vars=['RESOURCES'], var_name='TECHNOLOGIES', value_name='transportFactor' + k2)

        df2['dissipation'] = pd.concat(pd.DataFrame(
            data={'dissipation': [df_stransport.loc[(trans, 2020), 'dissipation']],
                  'RESOURCES': df_stransport.loc[(trans, 2020), 'resource'],
                  'TECHNOLOGIES': trans}) for trans in stranstechSet
        )
    transportFactors = pd.merge(
        df['charge'], df['discharge'], how='outer').fillna(0)
    transportFactors = pd.merge(transportFactors, df['dissipation'], how='outer').fillna(
        0).set_index(['RESOURCES', 'TECHNOLOGIES'])

    Calendrier = scenario['gridConnection']
    Economics = scenario['economicParameters'].melt(
        var_name='Eco').set_index('Eco')

    ResParameters = pd.concat((
        k.melt(id_vars=['TIMESTAMP', 'YEAR'], var_name=[
               'RESOURCES'], value_name=name).set_index(['YEAR', 'TIMESTAMP', 'RESOURCES'])
        for k, name in [(scenario['resourceImportPrices'], 'importCost'), (scenario['resourceImportCO2eq'], 'emission')]
    ), axis=1)

    availabilityFactor = scenario['availability']

    # Return hydrogen annual consumption in kt
    if printTables:
        print(areaConsumption.loc[slice(None), slice(
            None), 'electricity'].groupby('YEAR').sum()/33e3)
        print(TechParameters)
        print(CarbonTax)
        print(conversionFactor)
        print(StorageParameters)
        print(storageFactors)
        print(transportFactors)
        print(ResParameters)
        print(availabilityFactor)

    inputDict = scenario.copy()
    inputDict["areaConsumption"] = areaConsumption
    inputDict["availabilityFactor"] = availabilityFactor
    inputDict["techParameters"] = TechParameters
    inputDict["transportParameters"] = TransportParameters
    inputDict["transportFactors"] = transportFactors
    inputDict["resParameters"] = ResParameters
    inputDict["conversionFactor"] = conversionFactor
    inputDict["economics"] = Economics
    inputDict["calendar"] = Calendrier
    inputDict["storageParameters"] = StorageParameters
    inputDict["storageFactors"] = storageFactors
    inputDict["carbonTax"] = CarbonTax
    inputDict["transitionFactors"] = scenario["transitionFactors"]
    inputDict["yearList"] = scenario["yearList"]
    inputDict["areaList"] = scenario["areaList"]
    return inputDict

    # para globaux fin


def systemModelPedro(scenario, isAbstract=False):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :return: pyomo model
    """

    inputDict = loadScenario(scenario, False)

    yearList = np.array(inputDict["yearList"])
    dy = yearList[1] - yearList[0]
    y0 = yearList[0]

    areaList = np.array(inputDict["areaList"])

    # global
    areaConsumption = inputDict["areaConsumption"].loc[(
        inputDict["yearList"][1:], slice(None), slice(None), slice(None))]
    availabilityFactor = inputDict["availabilityFactor"].loc[(
        inputDict["yearList"][1:], slice(None), slice(None))]
    TechParameters = inputDict["techParameters"]
    TransportParameters = inputDict["transportParameters"]
    transportFactors = inputDict["transportFactors"]
    ResParameters = inputDict["resParameters"]
    conversionFactor = inputDict["conversionFactor"]
    Economics = inputDict["economics"]
    Calendrier = inputDict["calendar"]
    StorageParameters = inputDict["storageParameters"]
    storageFactors = inputDict["storageFactors"]
    TransFactors = inputDict["transitionFactors"]
    CarbonTax = inputDict["carbonTax"].loc[inputDict["yearList"][1:]]
    carbonGoals = inputDict["carbonGoals"].loc[inputDict["yearList"][1:]]
    inputDict["maxBiogasCap"] = inputDict["maxBiogasCap"].loc[inputDict["yearList"][1:]]

    isAbstract = False
    availabilityFactor.isna().sum()

    # Cleaning
    availabilityFactor = availabilityFactor.fillna(method='pad')
    areaConsumption = areaConsumption.fillna(method='pad')
    ResParameters = ResParameters.fillna(0)

    # obtaining dimensions values
    TECHNOLOGIES = set(
        TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO = set(
        StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    RESOURCES = set(ResParameters.index.get_level_values('RESOURCES').unique())
    TRANS_TECHNO = set(
        TransportParameters.index.get_level_values('TRANS_TECHNO').unique())
    RESOURCES = set(ResParameters.index.get_level_values('RESOURCES').unique())
    TIMESTAMP = set(
        areaConsumption.index.get_level_values('TIMESTAMP').unique())
    YEAR = set(yearList)

    # AREA
    AREA = set(areaList)

    TIMESTAMP_list = areaConsumption.index.get_level_values(
        'TIMESTAMP').unique()

    YEAR_list = yearList
    AREA_list = areaList  # puisque les zones sont uniques

    HORAIRE = {'P', 'HPH', 'HCH', 'HPE', 'HCE'}
    # Subsets
    TIMESTAMP_HCH = set(Calendrier[Calendrier['Calendrier']
                        == 'HCH'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_HPH = set(Calendrier[Calendrier['Calendrier']
                        == 'HPH'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_HCE = set(Calendrier[Calendrier['Calendrier']
                        == 'HCE'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_HPE = set(Calendrier[Calendrier['Calendrier']
                        == 'HPE'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_P = set(Calendrier[Calendrier['Calendrier']
                      == 'P'].index.get_level_values('TIMESTAMP').unique())

    #####################
    #    Pyomo model    #
    #####################

    if (isAbstract):
        model = pyomo.environ.AbstractModel()
    else:
        model = pyomo.environ.ConcreteModel()

    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.TRANS_TECHNO = Set(initialize=TRANS_TECHNO, ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO, ordered=False)
    model.RESOURCES = Set(initialize=RESOURCES, ordered=False)
    model.TIMESTAMP = Set(initialize=TIMESTAMP, ordered=False)
    model.YEAR = Set(initialize=YEAR, ordered=False)
    model.HORAIRE = Set(initialize=HORAIRE, ordered=False)
    model.YEAR_invest = Set(initialize=YEAR_list[:-1], ordered=False)
    model.YEAR_op = Set(initialize=YEAR_list[1:], ordered=False)
    model.AREA = Set(initialize=AREA, ordered=False)
    model.AREA_AREA = model.AREA * model.AREA
    model.YEAR_invest_TECHNOLOGIES = model.YEAR_invest*model.TECHNOLOGIES
    model.YEAR_invest_STOCKTECHNO = model.YEAR_invest * model.STOCK_TECHNO
    model.YEAR_op_TECHNOLOGIES = model.YEAR_op * model.TECHNOLOGIES
    model.YEAR_op_TIMESTAMP_TECHNOLOGIES = model.YEAR_op * \
        model.TIMESTAMP * model.TECHNOLOGIES
    model.YEAR_op_TIMESTAMP_STOCKTECHNO = model.YEAR_op * \
        model.TIMESTAMP * model.STOCK_TECHNO
    model.YEAR_invest_TRANSTECHNO = model.YEAR_invest * model.TRANS_TECHNO
    model.RESOURCES_TRANSTECHNO = model.RESOURCES * model.TRANS_TECHNO
    model.RESOURCES_TECHNOLOGIES = model.RESOURCES * model.TECHNOLOGIES
    model.RESOURCES_STOCKTECHNO = model.RESOURCES * model.STOCK_TECHNO
    model.YEAR_op_TIMESTAMP_RESOURCES_AREA = model.YEAR_op * \
        model.TIMESTAMP * model.RESOURCES * model.AREA
    model.TECHNOLOGIES_TECHNOLOGIES = model.TECHNOLOGIES*model.TECHNOLOGIES

    # Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(
        initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1], ordered=False)
    model.TIMESTAMP_MinusThree = Set(
        initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3], ordered=False)

    ###############
    # Parameters ##
    ###############

    # local
    model.areaConsumption = Param(model.YEAR_op_TIMESTAMP_RESOURCES_AREA, default=0,
                                  initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict(), domain=Reals)

    # global début
    model.availabilityFactor = Param(model.YEAR_op_TIMESTAMP_TECHNOLOGIES, domain=PercentFraction, default=1,
                                     initialize=availabilityFactor.loc[:, "availabilityFactor"].squeeze().to_dict())
    model.conversionFactor = Param(model.RESOURCES_TECHNOLOGIES, default=0,
                                   initialize=conversionFactor.loc[:, "conversionFactor"].squeeze().to_dict())
    model.carbon_goal = Param(model.YEAR_op, default=0, initialize=carbonGoals.loc[:, 'carbonGoals'].squeeze(
    ).to_dict(), domain=NonNegativeReals)
    model.carbon_taxe = Param(model.YEAR_op, default=0, initialize=CarbonTax.loc[:, 'carbonTax'].squeeze(
    ).to_dict(), domain=NonNegativeReals)
    #model.gazBio_max = Param(model.YEAR_op, default=0,initialize={2:103000,3:206000,4:310000}, domain=NonNegativeReals)
    model.gazBio_max = Param(model.YEAR_op, default=0, initialize=inputDict["maxBiogasCap"].loc[:, "maxBiogasCap"].squeeze(
    ).to_dict(), domain=NonNegativeReals)

    model.transFactor = Param(model.TECHNOLOGIES_TECHNOLOGIES, mutable=False,
                              default=0, initialize=TransFactors.loc[:, 'TransFactor'].squeeze().to_dict())

    gasTypes = ['biogas', 'natural gas']
    # with test of existing columns on TechParameters

    for COLNAME in TechParameters:
        # each column in TechParameters will be a parameter
        if COLNAME not in ["TECHNOLOGIES", "AREA", "YEAR"]:
            exec("model." + COLNAME + "= Param(model.YEAR_invest, model.TECHNOLOGIES, default=0, domain=Reals," +
                 "initialize=TechParameters." + COLNAME + ".loc[(inputDict['yearList'][:-1], slice(None))].squeeze().to_dict())")

    for COLNAME in ResParameters:
        # each column in TechParameters will be a parameter
        if COLNAME not in ["TECHNOLOGIES", "AREA", "YEAR"]:
            exec("model." + COLNAME + "= Param(model.YEAR_op_TIMESTAMP_RESOURCES, domain=NonNegativeReals,default=0," +
                 "initialize=ResParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in Calendrier:
        if COLNAME not in ["TIMESTAMP", "AREA"]:
            exec("model." + COLNAME + " = Param(model.TIMESTAMP, default=0," +
                 "initialize=Calendrier." + COLNAME + ".squeeze().to_dict(),domain=Any)")

    for COLNAME in StorageParameters:
        # each column in StorageParameters will be a parameter
        if COLNAME not in ["STOCK_TECHNO", "AREA", "YEAR"]:
            exec("model." + COLNAME + " =Param(model.YEAR_invest_STOCKTECHNO,domain=Any,default=0," +
                 "initialize=StorageParameters." + COLNAME + ".loc[(inputDict['yearList'][:-1], slice(None))].squeeze().to_dict())")

    for COLNAME in storageFactors:
        if COLNAME not in ["TECHNOLOGIES", "RESOURCES"]:
            exec("model." + COLNAME + " =Param(model.RESOURCES_STOCKTECHNO,domain=NonNegativeReals,default=0," +
                 "initialize=storageFactors." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in TransportParameters:
        # each column in StorageParameters will be a parameter
        if COLNAME not in ["TRANS_TECHNO", "AREA", "YEAR"]:
            exec("model." + COLNAME + " =Param(model.YEAR_invest_TRANSTECHNO,domain=Any,default=0," +
                 "initialize=TransportParameters." + COLNAME + ".loc[(inputDict['yearList'][:-1], slice(None))].squeeze().to_dict())")

    for COLNAME in transportFactors:
        if COLNAME not in ["TECHNOLOGIES", "RESOURCES"]:
            exec("model." + COLNAME + " =Param(model.RESOURCES_TRANSTECHNO,domain=NonNegativeReals,default=0," +
                 "initialize=transportFactors." + COLNAME + ".squeeze().to_dict())")

    # global fin

    ################
    # Variables    #
    ################

    # In this section, variables are separated in two categories : \
    # decision variables wich are the reals variables of the otimisation problem \
    # (these are noted Dvar), and problem variables which are resulting of calculation \
    # and are convenient for the readability and the analyse of results (these are noted Pvar)

    # Operation
    model.power_Dvar = Var(model.YEAR_op, model.TIMESTAMP, model.TECHNOLOGIES, model.AREA,
                           domain=NonNegativeReals)  # Power of a conversion mean at time t in the area 'area'
    model.importation_Dvar = Var(model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.AREA,
                                 domain=NonNegativeReals, initialize=0)  # Improtation of a resource at time t in the area 'area'
    # Amount of a resource at time t in the area 'area'
    model.energy_Pvar = Var(model.YEAR_op, model.TIMESTAMP,
                            model.RESOURCES, model.AREA)
    # Puissance souscrite max par plage horaire pour l'année d'opération y dans area
    model.max_PS_Dvar = Var(model.YEAR_op, model.HORAIRE, model.AREA,
                            domain=NonNegativeReals)
    # CO2 emission at each time t in area
    model.carbon_Pvar = Var(model.YEAR_op, model.TIMESTAMP, model.AREA)

    # Storage operation variables
    # level of the energy stock in a storage mean at time t in the area 'area'
    model.stockLevel_Pvar = Var(
        model.YEAR_op, model.TIMESTAMP, model.STOCK_TECHNO, model.AREA, domain=NonNegativeReals)
    model.storageIn_Pvar = Var(model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO, model.AREA,
                               domain=NonNegativeReals)  # Energy stored in a storage mean at time t
    model.storageOut_Pvar = Var(model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO, model.AREA,
                                domain=NonNegativeReals)  # Energy taken out of the in a storage mean at time t
    # Energy consumed the in a storage mean at time t (other than the one stored)
    model.storageConsumption_Pvar = Var(
        model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO, model.AREA, domain=NonNegativeReals)

    # Transport
    # Maximum transport flow from area a to b
    model.TmaxTot_Pvar = Var(
        model.YEAR_invest, model.TRANS_TECHNO, model.AREA_AREA, domain=Reals)    
    # New transport flow from area a to b
    model.TInvest_Dvar = Var(
        model.YEAR_invest, model.TRANS_TECHNO, model.AREA_AREA, domain=Reals)
    # Deleted transport flow from area a to b
    model.TDel_Dvar = Var(
        model.YEAR_invest, model.TRANS_TECHNO, model.AREA_AREA, domain=Reals)
    # Instant flow at time t from area a to b 
    model.FlowTot_Dvar = Var(
        model.YEAR_invest, model.TRANS_TECHNO, model.AREA_AREA, model.TIMESTAMP, domain=Reals)

    # Investment
    # Capacity of a conversion mean invested in year y in area 'area'
    model.capacityInvest_Dvar = Var(
        model.YEAR_invest, model.TECHNOLOGIES, model.AREA, domain=NonNegativeReals, initialize=0)
    # Capacity of a conversion mean that is removed each year y
    model.capacityDel_Pvar = Var(
        model.YEAR_invest, model.YEAR_invest, model.TECHNOLOGIES, model.AREA, domain=NonNegativeReals)
    # Transformation of technologies 1 into technologies 2
    model.transInvest_Dvar = Var(
        model.YEAR_invest, model.TECHNOLOGIES, model.TECHNOLOGIES, model.AREA, domain=NonNegativeReals)
    model.capacityDem_Dvar = Var(
        model.YEAR_invest, model.YEAR_invest, model.TECHNOLOGIES, model.AREA, domain=NonNegativeReals)
    model.capacity_Pvar = Var(
        model.YEAR_op, model.TECHNOLOGIES, model.AREA, domain=NonNegativeReals, initialize=0)
    # Maximum capacity of a storage mean
    model.CmaxInvest_Dvar = Var(
        model.YEAR_invest, model.STOCK_TECHNO, model.AREA, domain=NonNegativeReals)
    # Maximum flow of energy in/out of a storage mean
    model.PmaxInvest_Dvar = Var(
        model.YEAR_invest, model.STOCK_TECHNO, model.AREA, domain=NonNegativeReals)
    # Maximum capacity of a storage mean
    model.Cmax_Pvar = Var(
        model.YEAR_op, model.STOCK_TECHNO, model.AREA, domain=NonNegativeReals)
    # Maximum flow of energy in/out of a storage mean
    model.Pmax_Pvar = Var(
        model.YEAR_op, model.STOCK_TECHNO, model.AREA, domain=NonNegativeReals)
    model.CmaxDel_Dvar = Var(
        model.YEAR_invest, model.STOCK_TECHNO, model.AREA, domain=NonNegativeReals)
    model.PmaxDel_Dvar = Var(
        model.YEAR_invest, model.STOCK_TECHNO, model.AREA, domain=NonNegativeReals)

    # global ?? LOCAL
    # Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef

    # en gros c'est cb ça coûte par unité de temps (1 heure) de faire tourner telle techno dans telle ville à telle date à telle puissance
    model.powerCosts_Pvar = Var(model.YEAR_op, model.TECHNOLOGIES, model.AREA)
    # Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef
    model.capacityCosts_Pvar = Var(
        model.YEAR_op, model.TECHNOLOGIES, model.AREA)
    # Cost of ressource imported, explicitely defined by definition importCostsDef
    model.importCosts_Pvar = Var(model.YEAR_op, model.RESOURCES, model.AREA)
    # Coûts TURPE pour électricité
    model.turpeCosts_Pvar = Var(
        model.YEAR_op, model.RESOURCES, model.AREA, domain=NonNegativeReals)
    # Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
    model.storageCosts_Pvar = Var(
        model.YEAR_op, model.STOCK_TECHNO, model.AREA)
    # cost of CO2 emssion of one area
    model.carbonCosts_Pvar = Var(
        model.YEAR_op, model.AREA, domain=NonNegativeReals)


        

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    # locale
    def ObjectiveFunction_rule(model):  # OBJ
        return sum(
            sum(model.powerCosts_Pvar[y, tech, area] + model.capacityCosts_Pvar[y, tech, area]
                for tech in model.TECHNOLOGIES)
            + sum(model.importCosts_Pvar[y, res, area]
                  for res in model.RESOURCES)
            + sum(model.storageCosts_Pvar[y, s_tech, area]
                  for s_tech in STOCK_TECHNO)
            + model.turpeCosts_Pvar[y, 'electricity', area]
            + model.carbonCosts_Pvar[y, area]
            for y in model.YEAR_op for area in model.AREA) \
            + 0.5*sum(
            distance[area1, area2] * (
                sum(
                (model.transportPowerCost[y, ttech] + model.carbone_taxe * trans_emissions_co2[y, ttech]) * abs(
                    trans_energy[y, t, ttech, area1, area2])
                for t in model.TIMESTAMP)
            + (trans_capex[y, ttech] * f1(r, trans_lifetime[y -
               1, ttech]) + trans_opex[y, ttech]*f3(r, y)) * trans_Pvar[y, ttech, area1, area2]
               )
            for y in model.YEAR_op for ttech in model.TRANS_TECHNO for area1 in model.AREA for area2 in model.AREA
        )
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    # global
    r = Economics.loc['discountRate'].value
    i = Economics.loc['financeRate'].value

    def f1(r, n):  # This factor converts the investment costs into n annual repayments
        return r/((1+r)*(1-(1+r)**-n))

    def f3(r, y):  # This factor discounts a payment to y0 values
        return (1+r)**(-(y-y0))

    # powerCosts definition Constraints
    # on minimise à chaque instant et dans chaque zone, même si aucune dépendance selon la zone ?
    # EQ forall tech in TECHNOLOGIES powerCosts  = sum{t in TIMESTAMP} powerCost[tech]*power[t,tech] / 1E6;

    # global ???

    # par an et par techno ? faut-il mettre area en paramètre ou direct sommer dessus ??
    # on décide de le mettre en paramètre pour avoir les coûts par noeud
    def powerCostsDef_rule(model, y, tech, area):
        return sum(model.powerCost[y-dy, tech]*f3(r, y) * model.power_Dvar[y, t, tech, area] for t in model.TIMESTAMP)\
            == model.powerCosts_Pvar[y, tech, area]
    model.powerCostsCtr = Constraint(
        model.YEAR_op, model.TECHNOLOGIES, model.AREA, rule=powerCostsDef_rule)

    # capacityCosts definition Constraints
    # EQ forall tech in TECHNOLOGIES
    def capacityCostsDef_rule(model, y, tech, area):
        return sum(model.investCost[yi, tech]
                   * f1(i, model.lifeSpan[yi, tech]) * f3(r, y-dy)
                   * (model.capacityInvest_Dvar[yi, tech, area] - model.capacityDel_Pvar[yi, y-dy, tech, area])
                   for yi in yearList[yearList < y]) + model.operationCost[y-dy, tech]*f3(r, y)*model.capacity_Pvar[y, tech, area] == model.capacityCosts_Pvar[y, tech, area]
    model.capacityCostsCtr = Constraint(
        model.YEAR_op, model.TECHNOLOGIES, model.AREA, rule=capacityCostsDef_rule)

    # importCosts definition Constraints
    def importCostsDef_rule(model, y, res, area):
        return sum((model.importCost[y, t, res]*f3(r, y) * model.importation_Dvar[y, t, res, area])for t in model.TIMESTAMP) == model.importCosts_Pvar[y, res, area]
    model.importCostsCtr = Constraint(
        model.YEAR_op, model.RESOURCES, model.AREA, rule=importCostsDef_rule)

    # gaz definition Constraints
    def BiogazDef_rule(model, y, res):
        if res == 'biogas':
            return sum(model.importation_Dvar[y, t, res, area] for t in model.TIMESTAMP for area in model.AREA) <= model.gazBio_max[y]
        else:
            return Constraint.Skip
    model.BiogazCtr = Constraint(
        model.YEAR_op, model.RESOURCES, rule=BiogazDef_rule)

    # Carbon emission definition Constraints
    # c'est quoi model.emission ??
    def CarbonDef_rule(model, y, t, area):
        return sum((model.power_Dvar[y, t, tech, area] * model.EmissionCO2[y-dy, tech]) for tech in model.TECHNOLOGIES) + \
            sum(model.importation_Dvar[y, t, res, area]*model.emission[y, t, res]
                for res in model.RESOURCES) == model.carbon_Pvar[y, t, area]
    model.CarbonDefCtr = Constraint(
        model.YEAR_op, model.TIMESTAMP, rule=CarbonDef_rule)

    # def CarbonCtr_rule(model):
    # return sum(model.carbon_Pvar[y,t] for y,t in zip(model.YEAR_op,model.TIMESTAMP)) <= sum(model.carbon_goal[y] for y in model.YEAR_op)
    #model.CarbonCtr = Constraint(rule=CarbonCtr_rule)

    # def CarbonCtr_rule(model,y):
    #     return sum(model.carbon_Pvar[y,t] for t in model.TIMESTAMP) <= model.carbon_goal[y]
    # model.CarbonCtr = Constraint(model.YEAR_op,rule=CarbonCtr_rule)

    # CarbonCosts definition Constraint
    def CarbonCosts_rule(model, y, area):
        return model.carbonCosts_Pvar[y, area] == sum(model.carbon_Pvar[y, t, area]*model.carbon_taxe[y]*f3(r, y) for t in model.TIMESTAMP)
    model.CarbonCostsCtr = Constraint(
        model.YEAR_op, model.AREA, rule=CarbonCosts_rule)

    # TURPE
    def PuissanceSouscrite_rule(model, y, t, res, area):
        if res == 'electricity':
            if t in TIMESTAMP_P:
                # en MW
                return model.max_PS_Dvar[y, 'P', area] >= model.importation_Dvar[y, t, res, area]
            elif t in TIMESTAMP_HPH:
                return model.max_PS_Dvar[y, 'HPH', area] >= model.importation_Dvar[y, t, res, area]
            elif t in TIMESTAMP_HCH:
                return model.max_PS_Dvar[y, 'HCH', area] >= model.importation_Dvar[y, t, res, area]
            elif t in TIMESTAMP_HPE:
                return model.max_PS_Dvar[y, 'HPE', area] >= model.importation_Dvar[y, t, res, area]
            elif t in TIMESTAMP_HCE:
                return model.max_PS_Dvar[y, 'HCE', area] >= model.importation_Dvar[y, t, res, area]
        else:
            return Constraint.Skip
    model.PuissanceSouscriteCtr = Constraint(
        model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.AREA, rule=PuissanceSouscrite_rule)

    def TurpeCtr_rule(model, y, res, area):
        if res == 'electricity':
            return model.turpeCosts_Pvar[y, res, area] == (sum(model.HTA[t] * model.importation_Dvar[y, t, res, area] for t in TIMESTAMP) +
                                                           model.max_PS_Dvar[y, 'P', area]*16310+(model.max_PS_Dvar[y, 'HPH', area]-model.max_PS_Dvar[y, 'P', area])*15760+(model.max_PS_Dvar[y, 'HCH', area]-model.max_PS_Dvar[y, 'HPH', area])*13290+(model.max_PS_Dvar[y, 'HPE', area]-model.max_PS_Dvar[y, 'HCH', area])*8750+(model.max_PS_Dvar[y, 'HCE', area]-model.max_PS_Dvar[y, 'HPE', area])*1670)*f3(r, y)
        else:
            return model.turpeCosts_Pvar[y, res, area] == 0
    model.TurpeCtr = Constraint(
        model.YEAR_op, model.RESOURCES, model.AREA, rule=TurpeCtr_rule)

    # Capacity constraints selon les technologies choisies
    # clairement local
    # test multinoeud
    if ('CCS1' and 'CCS2') in model.TECHNOLOGIES:
        def capacityCCS_rule(model, y, tech, area):
            if tech == 'CCS1':
                return model.capacityInvest_Dvar[y, tech, area] == sum(model.transInvest_Dvar[y, tech1, tech2, area]
                                                                       for tech1, tech2 in [('SMR', 'SMR + CCS1'), ('SMR', 'SMR + CCS2'), ('Existing SMR', 'SMR + CCS1'), ('Existing SMR', 'SMR + CCS2')])
            elif tech == 'CCS2':
                return model.capacityInvest_Dvar[y, tech, area] == sum(model.transInvest_Dvar[y, tech1, tech2, area]
                                                                       for tech1, tech2 in [('SMR', 'SMR + CCS2'), ('Existing SMR', 'SMR + CCS2'), ('SMR + CCS1', 'SMR + CCS2')])
            else:
                return Constraint.Skip
        model.capacityCCSCtr = Constraint(
            model.YEAR_invest, model.TECHNOLOGIES, model.AREA, rule=capacityCCS_rule)

    def TransInvest_rule(model, y, tech1, tech2, area):
        if model.transFactor[tech1, tech2] == 0:
            return model.transInvest_Dvar[y, tech1, tech2, area] == 0
        else:
            return Constraint.Skip
    model.TransInvestCtr = Constraint(
        model.YEAR_invest, model.TECHNOLOGIES, model.TECHNOLOGIES, model.AREA, rule=TransInvest_rule)

    if 'Existing SMR' in model.TECHNOLOGIES:
        def TransCapacity_rule(model, y, tech, area):
            if y == y0:
                return sum(model.transInvest_Dvar[y, 'Existing SMR', tech2, area] for tech2 in model.TECHNOLOGIES) <= model.capacityInvest_Dvar[y, 'Existing SMR', area]
            else:
                return sum(model.transInvest_Dvar[y, tech, tech2, area] for tech2 in model.TECHNOLOGIES) <= model.capacity_Pvar[y, tech, area]
        model.TransCapacityCtr = Constraint(
            model.YEAR_invest, model.TECHNOLOGIES, model.AREA, rule=TransCapacity_rule)

    def CapacityDemUB_rule(model, yi, y, tech, area):
        if yi == model.yearStart[y, tech]:
            return sum(model.capacityDem_Dvar[yi, z, tech, area] for z in yearList[yearList <= y]) == model.capacityInvest_Dvar[yi, tech, area]
        elif yi > y:
            return model.capacityDem_Dvar[yi, y, tech, area] == 0
        else:
            return sum(model.capacityDem_Dvar[yi, yt, tech, area] for yt in model.YEAR_invest) <= model.capacityInvest_Dvar[yi, tech, area]
    model.CapacityDemUBCtr = Constraint(
        model.YEAR_invest, model.YEAR_invest, model.TECHNOLOGIES, model.AREA, rule=CapacityDemUB_rule)

    # def CapacityDemUP_rule(model,y, tech):
    #     if y == 1:
    #         return Constraint.Skip
    #     else :
    #         if tech in ['SMR_class','SMR_class_ex','SMR_elec'] :
    #             return sum(model.capacityDem_Dvar[yi,y,tech] for yi in model.YEAR_invest) >= model.capacity_Pvar[y,tech] - sum(model.power_Dvar[y,t,tech] for t in TIMESTAMP)/8760/0.2
    #         elif tech in ['SMR_CCS1','SMR_CCS2','SMR_elecCCS1'] :
    #             return sum(model.capacityDem_Dvar[yi, y, tech] for yi in model.YEAR_invest) >= model.capacity_Pvar[y,tech] - sum(model.power_Dvar[y,t, tech] for t in TIMESTAMP) / 8760 / 0.5
    #         else :
    #             return Constraint.Skip
    # model.CapacityDemUPCtr = Constraint(model.YEAR_invest,model.TECHNOLOGIES, rule=CapacityDemUP_rule)

    def CapacityDel_rule(model, yi, y, tech, area):
        if model.yearStart[y, tech] >= yi:
            return model.capacityDel_Pvar[yi, y, tech, area] == model.capacityInvest_Dvar[yi, tech, area]
        else:
            return model.capacityDel_Pvar[yi, y, tech, area] == 0
    model.CapacityDelCtr = Constraint(
        model.YEAR_invest, model.YEAR_invest, model.TECHNOLOGIES, model.AREA, rule=CapacityDel_rule)

    def CapacityTot_rule(model, y, tech, area):
        if y == y0+dy:
            return model.capacity_Pvar[y, tech, area] == model.capacityInvest_Dvar[y-dy, tech, area] - \
                sum(model.capacityDem_Dvar[yi, y-dy, tech, area] for yi in model.YEAR_invest) + sum(model.transInvest_Dvar[y-dy, tech1, tech, area]
                                                                                                    for tech1 in model.TECHNOLOGIES) - sum(model.transInvest_Dvar[y-dy, tech, tech2, area] for tech2 in model.TECHNOLOGIES)
        else:
            return model.capacity_Pvar[y, tech, area] == model.capacity_Pvar[y-dy, tech, area] - \
                sum(model.capacityDem_Dvar[yi, y-dy, tech, area] for yi in model.YEAR_invest) + \
                model.capacityInvest_Dvar[y-dy, tech, area] + sum(model.transInvest_Dvar[y-dy, tech1, tech, area]
                                                                  for tech1 in model.TECHNOLOGIES) - sum(model.transInvest_Dvar[y-dy, tech, tech2, area] for tech2 in model.TECHNOLOGIES)
    model.CapacityTotCtr = Constraint(
        model.YEAR_op, model.TECHNOLOGIES, model.AREA, rule=CapacityTot_rule)

    def Capacity_rule(model, y, t, tech, area):  # INEQ forall t, tech
        return model.capacity_Pvar[y, tech, area] * model.availabilityFactor[y, t, tech] >= model.power_Dvar[y, t, tech, area]
    model.CapacityCtr = Constraint(
        model.YEAR_op, model.TIMESTAMP, model.TECHNOLOGIES, model.AREA, rule=Capacity_rule)

    # Ressource production constraint
    # local
    def Production_rule(model, y, t, res, area):  # EQ forall t, res
        if res == 'gas':
            return sum(model.power_Dvar[y, t, tech, area] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) + sum(model.importation_Dvar[y, t, resource, area] for resource in gasTypes) + \
                sum(model.storageOut_Pvar[y, t, res, s_tech, area] - model.storageIn_Pvar[y, t, res, s_tech, area] -
                    model.storageConsumption_Pvar[y, t, res, s_tech, area] for s_tech in STOCK_TECHNO) == model.energy_Pvar[y, t, res, area]
        elif res in gasTypes:
            return model.energy_Pvar[y, t, res, area] == 0
        else:
            return sum(model.power_Dvar[y, t, tech, area] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) + \
                model.importation_Dvar[y, t, res, area] + sum(model.storageOut_Pvar[y, t, res, s_tech, area] - model.storageIn_Pvar[y, t, res, s_tech, area] -
                                                              model.storageConsumption_Pvar[y, t, res, s_tech, area] for s_tech in STOCK_TECHNO) == model.energy_Pvar[y, t, res, area]
    model.ProductionCtr = Constraint(
        model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.AREA, rule=Production_rule)

    # contrainte d'equilibre offre demande
    # local
    def energyCtr_rule(model, y, t, res, area):  # INEQ forall t
        return model.energy_Pvar[y, t, res, area] == model.areaConsumption[y, t, res, area]
    model.energyCtr = Constraint(
        model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.AREA, rule=energyCtr_rule)

    # Storage power and capacity constraints
    # local
    def StorageCmaxTot_rule(model, y, stech, area):  # INEQ forall t, tech
        if y == y0+dy:
            return model.Cmax_Pvar[y, stech, area] == model.CmaxInvest_Dvar[y-dy, stech, area] - model.CmaxDel_Dvar[y-dy, stech, area]
        else:
            return model.Cmax_Pvar[y, stech, area] == model.Cmax_Pvar[y-dy, stech, area] + model.CmaxInvest_Dvar[y-dy, stech, area] - model.CmaxDel_Dvar[y-dy, stech, area]
    model.StorageCmaxTotCtr = Constraint(
        model.YEAR_op, model.STOCK_TECHNO, model.AREA, rule=StorageCmaxTot_rule)

    def StoragePmaxTot_rule(model, y, s_tech, area):  # INEQ forall t, tech
        if y == y0+dy:
            return model.Pmax_Pvar[y, s_tech, area] == model.PmaxInvest_Dvar[y-dy, s_tech] - model.PmaxDel_Dvar[y-dy, s_tech]
        else:
            return model.Pmax_Pvar[y, s_tech, area] == model.Pmax_Pvar[y-dy, s_tech, area] + model.PmaxInvest_Dvar[y-dy, s_tech, area] - model.PmaxDel_Dvar[y-dy, s_tech, area]
    model.StoragePmaxTotCtr = Constraint(
        model.YEAR_op, model.STOCK_TECHNO, model.AREA, rule=StoragePmaxTot_rule)

    # storageCosts definition Constraint
    # EQ forall s_tech in STOCK_TECHNO
    def storageCostsDef_rule(model, y, s_tech, area):
        return sum((model.storageEnergyCost[yi, s_tech] * model.Cmax_Pvar[yi+dy, s_tech, area] +
                    model.storagePowerCost[yi, s_tech] * model.Pmax_Pvar[yi+dy, s_tech, area]) * f1(i, model.storagelifeSpan[yi, s_tech]) * f3(r, y-dy) for yi in yearList[yearList < y]) \
            + model.storageOperationCost[y-dy, s_tech]*f3(
                r, y) * model.Pmax_Pvar[y, s_tech, area] == model.storageCosts_Pvar[y, s_tech, area]
    model.storageCostsCtr = Constraint(
        model.YEAR_op, model.STOCK_TECHNO, model.AREA, rule=storageCostsDef_rule)

    # Storage max capacity constraint
    def storageCapacity_rule(model, y, s_tech, area):  # INEQ forall s_tech
        return model.CmaxInvest_Dvar[y, s_tech, area] <= model.c_max[y, s_tech]
    model.storageCapacityCtr = Constraint(
        model.YEAR_invest, model.STOCK_TECHNO, rule=storageCapacity_rule)

    def storageCapacityDel_rule(model, y, stech, area):
        if model.storageYearStart[y, stech] > 0:
            return model.CmaxDel_Dvar[y, stech, area] == model.CmaxInvest_Dvar[model.storageYearStart[y, stech], stech, area]
        else:
            return model.CmaxDel_Dvar[y, stech, area] == 0
    model.storageCapacityDelCtr = Constraint(
        model.YEAR_invest, model.STOCK_TECHNO, model.AREA, rule=storageCapacityDel_rule)

    # Storage max power constraint
    def storagePower_rule(model, y, s_tech, area):  # INEQ forall s_tech
        return model.PmaxInvest_Dvar[y, s_tech, area] <= model.p_max[y, s_tech]
    model.storagePowerCtr = Constraint(
        model.YEAR_invest, model.STOCK_TECHNO, model.AREA, rule=storagePower_rule)

    # contraintes de stock puissance
    def StoragePowerUB_rule(model, y, t, res, s_tech, area):  # INEQ forall t
        if res == model.resource[y-dy, s_tech]:
            return model.storageIn_Pvar[y, t, res, s_tech, area] - model.Pmax_Pvar[y, s_tech, area] <= 0
        else:
            return model.storageIn_Pvar[y, t, res, s_tech, area] == 0
    model.StoragePowerUBCtr = Constraint(
        model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO, model.AREA, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model, y, t, res, s_tech, area):  # INEQ forall t
        if res == model.resource[y-dy, s_tech]:
            return model.storageOut_Pvar[y, t, res, s_tech, area] - model.Pmax_Pvar[y, s_tech, area] <= 0
        else:
            return model.storageOut_Pvar[y, t, res, s_tech, area] == 0
    model.StoragePowerLBCtr = Constraint(
        model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO, model.AREA, rule=StoragePowerLB_rule)

    def storagePowerDel_rule(model, y, stech, area):
        if model.storageYearStart[y, stech] > 0:
            return model.PmaxDel_Dvar[y, stech, area] == model.PmaxInvest_Dvar[model.storageYearStart[y, stech], stech, area]
        else:
            return model.PmaxDel_Dvar[y, stech, area] == 0
    model.storagePowerDelCtr = Constraint(
        model.YEAR_invest, model.STOCK_TECHNO, model.AREA, rule=storagePowerDel_rule)

    # contrainte de consommation du stockage (autre que l'énergie stockée)
    def StorageConsumption_rule(model, y, t, res, s_tech, area):  # EQ forall t
        temp = model.resource[y-dy, s_tech]
        if res == temp:
            return model.storageConsumption_Pvar[y, t, res, s_tech, area] == 0
        else:
            return model.storageConsumption_Pvar[y, t, res, s_tech, area] == model.storageFactorIn[res, s_tech] * \
                model.storageIn_Pvar[y, t, temp, s_tech, area] + model.storageFactorOut[res, s_tech] * model.storageOut_Pvar[
                y, t, temp, s_tech, area]
    model.StorageConsumptionCtr = Constraint(
        model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO, model.AREA, rule=StorageConsumption_rule)

    # contraintes de stock capacité
    def StockLevel_rule(model, y, t, s_tech, area):  # EQ forall t
        res = model.resource[y-dy, s_tech]
        if t > 1:
            return model.stockLevel_Pvar[y, t, s_tech, area] == model.stockLevel_Pvar[y, t - 1, s_tech, area] * (
                1 - model.dissipation[res, s_tech]) + model.storageIn_Pvar[y, t, res, s_tech, area] * \
                model.storageFactorIn[res, s_tech] - model.storageOut_Pvar[y, t, res, s_tech, area] * model.storageFactorOut[
                res, s_tech]
        else:
            return model.stockLevel_Pvar[y, t, s_tech, area] == model.stockLevel_Pvar[y, 8760, s_tech, area] + model.storageIn_Pvar[y, t, res, s_tech, area] * \
                model.storageFactorIn[res, s_tech] - model.storageOut_Pvar[y, t, res, s_tech, area] * model.storageFactorOut[
                res, s_tech]
    model.StockLevelCtr = Constraint(
        model.YEAR_op, model.TIMESTAMP, model.STOCK_TECHNO, model.AREA, rule=StockLevel_rule)

    def StockCapacity_rule(model, y, t, s_tech, area):  # INEQ forall t
        return model.stockLevel_Pvar[y, t, s_tech, area] <= model.Cmax_Pvar[y, s_tech, area]
    model.StockCapacityCtr = Constraint(
        model.YEAR_op, model.TIMESTAMP, model.STOCK_TECHNO, model.AREA, rule=StockCapacity_rule)

    if "capacityLim" in TechParameters:
        def capacityLim_rule(model, y, tech, area):  # INEQ forall t, tech
            return model.capacityLim[y, tech] >= model.capacity_Pvar[y+dy, tech, area]
        model.capacityLimCtr = Constraint(
            model.YEAR_invest, model.TECHNOLOGIES, model.AREA, rule=capacityLim_rule)

    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model, y, tech, area):  # INEQ forall t, tech
            return model.maxCapacity[y, tech] >= model.capacityInvest_Dvar[y, tech, area]
        model.maxCapacityCtr = Constraint(
            model.YEAR_invest, model.TECHNOLOGIES, model.AREA, rule=maxCapacity_rule)

    if "minCapacity" in TechParameters:
        def minCapacity_rule(model, y, tech, area):  # INEQ forall t, tech
            return model.minCapacity[y, tech] <= model.capacityInvest_Dvar[y, tech, area]
        model.minCapacityCtr = Constraint(
            model.YEAR_invest, model.TECHNOLOGIES, model.AREA, rule=minCapacity_rule)

    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model, y, tech, area):  # INEQ forall t, tech
            if model.EnergyNbhourCap[y-dy, tech] > 0:
                return model.EnergyNbhourCap[y-dy, tech] * model.capacity_Pvar[y, tech, area] >= sum(
                    model.power_Dvar[y, t, tech, area] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip
        model.storageCtr = Constraint(
            model.YEAR_op, model.TECHNOLOGIES, model.AREA, rule=storage_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model, y, t, tech, area):  # INEQ forall t<
            if model.RampConstraintPlus[y-dy, tech] > 0:
                return model.power_Dvar[y, t+1, tech, area] - model.power_Dvar[y, t, tech, area] <= model.capacity_Pvar[y, tech, area] * model.RampConstraintPlus[y-dy, tech]
            else:
                return Constraint.Skip
        model.rampCtrPlus = Constraint(
            model.YEAR_op, model.TIMESTAMP_MinusOne, model.TECHNOLOGIES, model.AREA, rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model, y, t, tech, area):  # INEQ forall t<
            if model.RampConstraintMoins[y-dy, tech] > 0:
                var = model.power_Dvar[y, t + 1, tech, area] - \
                    model.power_Dvar[y, t, tech, area]
                return var >= - model.capacity_Pvar[y, tech, area] * model.RampConstraintMoins[y-dy, tech]
            else:
                return Constraint.Skip
        model.rampCtrMoins = Constraint(
            model.YEAR_op, model.TIMESTAMP_MinusOne, model.TECHNOLOGIES, model.AREA, rule=rampCtrMoins_rule)

    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model, y, t, tech, area):  # INEQ forall t<
            if model.RampConstraintPlus2[y-dy, tech] > 0:
                var = (model.power_Dvar[y, t + 2, tech, area] + model.power_Dvar[y, t + 3, tech, area]) / 2 - (
                    model.power_Dvar[y, t + 1, tech, area] + model.power_Dvar[y, t, tech, area]) / 2
                return var <= model.capacity_Pvar[y, tech, area] * model.RampConstraintPlus[y-dy, tech]
            else:
                return Constraint.Skip
        model.rampCtrPlus2 = Constraint(
            model.YEAR_op, model.TIMESTAMP_MinusThree, model.TECHNOLOGIES, model.AREA, rule=rampCtrPlus2_rule)

    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model, y, t, tech, area):  # INEQ forall t<
            if model.RampConstraintMoins2[y-dy, tech] > 0:
                var = (model.power_Dvar[y, t + 2, tech, area] + model.power_Dvar[y, t + 3, tech, area]) / 2 - (
                    model.power_Dvar[y, t + 1, tech, area] + model.power_Dvar[y, t, tech, area]) / 2
                return var >= - model.capacity_Pvar[y, tech, area] * model.RampConstraintMoins2[y-dy, tech]
            else:
                return Constraint.Skip
        model.rampCtrMoins2 = Constraint(
            model.YEAR_op, model.TIMESTAMP_MinusThree, model.TECHNOLOGIES, model.AREA, rule=rampCtrMoins2_rule)

    # Contraintes sur le transport.

    return model
