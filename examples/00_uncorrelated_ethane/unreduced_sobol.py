# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Demo Notebook for Sobol Sampling without Model Reduction

# +
import os
import yaml
import subprocess
import cantera as ct

import rmgpy.chemkin
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import ezuq.sobol
import ezuq.morris_screen
import ezuq.util
import ezuq.simulation
# -

working_dir = os.path.abspath('.')
gas = ct.Solution(os.path.join(working_dir, 'chem_annotated.yaml'))

# +
# Specify conditions for the simulation
temperatures = [800, 900, 1000]

i_C2H6 = ezuq.util.get_i_thing_ct({'C': 2, 'H': 6}, gas)
i_O2 = ezuq.util.get_i_thing_ct({'O': 2}, gas)
i_Ar = ezuq.util.get_i_thing_ct({'Ar': 1}, gas)  # He for Nancy data
x_C2H6 = 0.18
x_O2 = 0.64
x_Ar = 1.0 - x_C2H6 - x_O2
X = f'{gas.species_names[i_C2H6]}: {x_C2H6}, {gas.species_names[i_O2]}: {x_O2}, {gas.species_names[i_Ar]}: {x_Ar}'

conditions = []
for T in temperatures:
    conditions.append({
        'name': f'unreduced_{T}K',        # name for the directory where analysis will be run
        'temperature': T,       # K
        'pressure': ct.one_atm, # Pa
        'composition': X,
        'residence_time': 6.0,  # s
        'volume': 9.5e-5,       # m^3
    })

# Here we reduce the number of samples for much faster runtime, but you'll probably want to do ~100
ezuq.sobol.setup_runfiles(working_dir, conditions, morris_dir=None, i_sens=14)
# -


