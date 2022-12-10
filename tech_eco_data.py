from scipy.interpolate import interp1d
import numpy as np

def get_biogas_share_in_network_RTE(year): 
	return np.interp(year, [2019, 2030, 2040, 2050], [0] * 4)#[.001, .11, .37, 1])

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

	elif tech == "ElectrolysisS":
			capex = {
				'ref':  interp1d(years, electrolyser_capex_Reksten2022(tech='PEM', Pel=1, year=np.array(years))),
			}
			opex = {
				'ref': interp1d(years, [12] * 5),
			}
			life = {
				'ref':  interp1d(years, [30] * 5),
			}

	elif tech == "ElectrolysisM":
			capex = {
				'ref':  interp1d(years, electrolyser_capex_Reksten2022(tech='PEM', Pel=10, year=np.array(years))),
			}
			opex = {
				'ref': interp1d(years, [12] * 5),
			}
			life = {
				'ref':  interp1d(years, [30] * 5),
			}

	elif tech == "ElectrolysisL":
			capex = {
				'ref':  interp1d(years, electrolyser_capex_Reksten2022(tech='PEM', Pel=100, year=np.array(years))),
			}
			opex = {
				'ref': interp1d(years, [12] * 5),
			}
			life = {
				'ref':  interp1d(years, [30] * 5),
			}


	elif tech == 'Battery - 1h': 
			capex = {
				'ref':  interp1d(years, [537, 406, 332, 315, 315]), # EUR/kW
			}		

			opex = {
				'ref':  interp1d(years, [11] * 5), # EUR/kW/yr
			}	
			life = {
				'ref':  interp1d(years, [15] * 5),
			}

	elif tech == 'Battery - 4h': 
			capex = {
				'ref':  interp1d(years, [1480, 1101, 855, 740, 740]), # EUR/kW
			}		

			opex = {
				'ref':  interp1d(years, [30] * 5), # EUR/kW/yr
			}	
			life = {
				'ref':  interp1d(years, [15] * 5),
			}


	elif tech == 'Salt cavern': 
			capex = {
				'ref':  interp1d(years, [350] * 5), # EUR/kWhLHV
			}		

			opex = {
				'ref':  interp1d(years, [2] * 5), # EUR/kW/yr
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

def electrolyser_capex_Reksten2022(tech='PEM', Pel=100, year=2020):
	'''
	Reference: Reksten et al. (2022) https://www.sciencedirect.com/science/article/pii/S0360319922040253
	Pel: electrolyser electrical power in kW
	tech: electrolyser technology
	year: installation year 
	'''
	if tech=='PEM':
		alpha, beta, k0, k = 0.622, -158.9, 585.85, 9458.2
	elif tech=='Alkaline': 
		alpha, beta, k0, k = 0.649, -27.33, 301.04, 11603

	return (k0 + k/Pel * Pel**alpha) * (year/2020) ** beta