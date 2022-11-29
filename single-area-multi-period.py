import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time 
import os 

from Functions.f_multiResourceModelsPedro_copy import systemModelPedro
from Functions.f_optimization import getVariables_panda, getConstraintsDual_panda

from scenarios import scenario


solver= 'mosek'

print('Building model...')
model = systemModelPedro(scenario,isAbstract=False)

start_clock = time.time()
print('Calculating...')
opt = SolverFactory(solver)
results = opt.solve(model)
end_clock = time.time()
print('Computational time: {:.0f} s'.format(end_clock - start_clock)) 

res = {
    'variables': getVariables_panda(model), 
}

try: 
    res['constraints'] = getConstraintsDual_panda(model)
except KeyError: # This exception will be raised for a MILP case
    pass 

outputFolder = 'out'

try: 
    os.mkdir(outputFolder)
except: 
    pass

for v in res['variables'].keys():
    print(v)

for k, v in res['variables'].items():
    print ('Writing ' + k + '...') 
    v.to_csv(outputFolder + '/' + k + '.csv',index=True)

print(res['variables']['capacity_Pvar'])