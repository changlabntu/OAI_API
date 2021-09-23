import numpy as np
from main_oaimeta import MOAKS_get_vars
import os
import pandas as pd

df00 = pd.read_csv('extracted_meta/df00.csv')
v00 = pd.read_csv('extracted_meta/path00m.csv')
v00['cohort'] = [x.split('/')[0] for x in v00['folders']]
v00['knee'] = [x for x in zip(v00['ID'], v00['sequences'])]

moaks_vars = pd.read_excel('extracted_meta/KMRI_SQ_MOAKS_variables_summary.xlsx',
                           engine='openpyxl')

bml = MOAKS_get_vars(moaks_vars, ['BML Size'], ver='00')
bmln = MOAKS_get_vars(moaks_vars, ['BML #'], ver='00')
eff = MOAKS_get_vars(moaks_vars, ['Whole Knee Effusion'], ver='00')

i = 0
subjects = df00.loc[df00[eff[0]] == i, ['ID', 'SIDE'] + eff]
list_TSE = [(x[0], ['SAG_IW_TSE_RIGHT', 'SAG_IW_TSE_LEFT'][x[1] - 1]) for x in
                     zip(subjects['ID'], subjects['SIDE'])]
list_DESS = [(x[0], ['SAG_3D_DESS_RIGHT', 'SAG_3D_DESS_LEFT'][x[1] - 1]) for x in
                     zip(subjects['ID'], subjects['SIDE'])]
total = v00.loc[v00['knee'].isin(list_TSE + list_DESS), :]

total.to_csv('OAI00eff0.csv')