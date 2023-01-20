from scipy.interpolate import interp1d
import numpy as np

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
power_S = 600e-3  # MW

# liste des puissances max de ressources
# - pour une tech : produites par une installation (en MW)
# - pour une ttech : transportables par km (en MW/km)
p_max_fonc = {
    "Offshore wind - floating": 12,
    "Onshore wind": 6,
    "Ground PV": 6.4*10**(-5),
    "ElectrolysisS": power_S,
    "ElectrolysisM": 10 * power_S,
    "ElectrolysisL": 100 * power_S,
    # puissance maximale de fonctionnement du pipeline (=débit max), fixée
    "Pipeline_S": 100,
    "Pipeline_M": 1000,
    "Pipeline_L": 10000,
    # capacité d'un camion : C = 600kg = 600*33 = 19800 kWh = 19.8 MWh
    # c'est aussi la qté d'hydrogène qu'un camion transport en 1 heure sur 1 km
    "truckTransportingHydrogen": 19.8
}


def get_biogas_share_in_network_RTE(year):
    # [.001, .11, .37, 1])
    return np.interp(year, [2019, 2030, 2040, 2050], [0] * 4)


def get_capex_new_tech_RTE(tech, hyp='ref', year=2020, var=None):
    # https://assets.rte-france.com/prod/public/2022-06/FE2050%20_Rapport%20complet_ANNEXES.pdf page 937
    years = [2020, 2030, 2040, 2050, 2060]

    if tech == "Offshore wind - floating":
        capex = {
            'ref':  interp1d(years, [3100, 2500, 2200, 1900, 1900]),
            'low':  interp1d(years, [3100, 2100, 1700, 1300, 1300]),
            'high': interp1d(years, [3100, 2900, 2700, 2500, 2500]),
        }
        opex = {
            'high': interp1d(years, [110, 90, 80, 70, 70]),
            'low': interp1d(years,  [110, 75, 50, 40, 40]),
            'ref': interp1d(years,  [110, 80, 60, 50, 50]),
        }
        life = {
            'high':  interp1d(years, [20, 25, 30, 40, 40]),
            'low':  interp1d(years, [20, 25, 30, 40, 40]),
            'ref':  interp1d(years, [20, 25, 30, 40, 40]),
        }

    elif tech == "Onshore wind":
        capex = {
            'ref':  interp1d(years, [1300, 1200, 1050, 900, 900]),
            'low':  interp1d(years, [1300, 710, 620, 530, 530]),
            'high': interp1d(years, [1300, 1300, 1300, 1300, 1300]),
        }
        opex = {
            'high': interp1d(years, [40, 40, 40, 40, 40]),
            'low': interp1d(years,  [40, 22, 18, 16, 16]),
            'ref': interp1d(years,  [40, 35, 30, 25, 25]),
        }
        life = {
            'high':  interp1d(years, [25, 30, 30, 30, 30]),
            'low':  interp1d(years, [25, 30, 30, 30, 30]),
            'ref':  interp1d(years, [25, 30, 30, 30, 30]),
        }

    elif tech == "Ground PV":
        capex = {
            'ref':  interp1d(years, [747, 597, 517, 477, 477]),
            'low':  interp1d(years, [747, 557, 497, 427, 427]),
            'high': interp1d(years, [747, 612, 562, 527, 527]),
        }
        opex = {
            'high': interp1d(years, [11, 10, 10, 9, 9]),
            'low': interp1d(years,  [11, 9, 8, 7, 7]),
            'ref': interp1d(years,  [11, 10, 9, 8, 8]),
        }
        life = {
            'high':  interp1d(years, [25, 30, 30, 30, 30]),
            'low':  interp1d(years, [25, 30, 30, 30, 30]),
            'ref':  interp1d(years, [25, 30, 30, 30, 30]),
        }

    # modification des puissances p_max_fonc selon les données de scenarios.py
    # comme Pel est ici la puissance électrique consommée, on divise par le facteur de conversion élec->H
    elif tech == "ElectrolysisS":
        capex = {
            'ref':  interp1d(years, electrolyser_capex_Reksten2022(tech='PEM', Pel=p_max_fonc[tech] / conv_el_h, year=np.array(years)
                                                                   )
                             ),
        }
        opex = {
            'ref': interp1d(years, [12] * 5),
        }
        life = {
            'ref':  interp1d(years, [30] * 5),
        }

    elif tech == "ElectrolysisM":
        capex = {
            'ref':  interp1d(years, electrolyser_capex_Reksten2022(tech='PEM', Pel=p_max_fonc[tech] / conv_el_h, year=np.array(years))
                             ),
        }
        opex = {
            'ref': interp1d(years, [12] * 5),
        }
        life = {
            'ref':  interp1d(years, [30] * 5),
        }

    elif tech == "ElectrolysisL":
        capex = {
            'ref':  interp1d(
                years,
                electrolyser_capex_Reksten2022(
                    tech='PEM',
                    Pel=p_max_fonc[tech] / conv_el_h,
                    year=np.array(years)
                )
            ),
        }
        opex = {
            'ref': interp1d(years, [12] * 5),
        }
        life = {
            'ref':  interp1d(years, [30] * 5),
        }

    elif tech == 'Battery - 1h':
        capex = {
            'ref':  interp1d(years, [537, 406, 332, 315, 315]),  # EUR/kW
        }

        opex = {
            'ref':  interp1d(years, [11] * 5),  # EUR/kW/yr
        }
        life = {
            'ref':  interp1d(years, [15] * 5),
        }

    elif tech == 'Battery - 4h':
        capex = {
            'ref':  interp1d(years, [1480, 1101, 855, 740, 740]),  # EUR/kW
        }

        opex = {
            'ref':  interp1d(years, [30] * 5),  # EUR/kW/yr
        }
        life = {
            'ref':  interp1d(years, [15] * 5),
        }

    elif tech == 'Salt cavern':
        capex = {
            'ref':  interp1d(years, [350] * 5),  # EUR/kWhLHV
        }

        opex = {
            'ref':  interp1d(years, [2] * 5),  # EUR/kW/yr
        }
        life = {
            'ref':  interp1d(years, [40] * 5),
        }

    if var == "capex":
        return 1e3 * capex[hyp](year)
    elif var == "opex":
        return 1e3 * opex[hyp](year)
    elif var == 'lifetime':
        return life[hyp](year)
    else:
        return 1e3 * capex[hyp](year), 1e3 * opex[hyp](year), float(life[hyp](year))


def electrolyser_capex_Reksten2022(tech, Pel, year=2020):
    '''
    Reference: Reksten et al. (2022) https://www.sciencedirect.com/science/article/pii/S0360319922040253

        Pel: electrolyser electrical power consumption (MW)
    tech: electrolyser technology
    year: installation year 
    '''
    # conversion des MW en kW, unité dans laquelle la formule est écrite
    pel_kW = Pel * 1.e3

    if tech == 'PEM':
        alpha, beta, k0, k = 0.622, -158.9, 585.85, 9458.2
    elif tech == 'Alkaline':
        alpha, beta, k0, k = 0.649, -27.33, 301.04, 11603

    return (k0 + k/pel_kW * pel_kW**alpha) * (year/2020) ** beta

# capex/kW pour electrolyseur S en 2050 = 748.70€
# M : 654€
# L : 614.41€