import numpy as np
import pandas as pd
import os
from functools import reduce


def MOAKS_get_vars(moaks_summary, categories, ver):
    moaks_variables = moaks_summary.loc[moaks_summary['CATEGORY'].isin(categories), 'VARIABLE']
    l = list(moaks_variables.values)
    return [x.replace('$$', ver) for x in l]

