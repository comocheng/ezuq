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

# # Demo of Morris Screening.
#
# This truncates in a way that respects correlations

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

import ezuq.monte_carlo
import ezuq.morris_screen
import ezuq.util

import SALib.analyze.morris

# +
working_dir = os.path.abspath('.')
gas = ct.Solution(os.path.join(working_dir, 'chem_annotated.yaml'))

i_sens = 14

chemkin_file = os.path.join(working_dir, 'chem_annotated.inp')
dictionary_file = os.path.join(working_dir, 'species_dictionary.txt')
species_list, reaction_list = rmgpy.chemkin.load_chemkin_file(chemkin_file, dictionary_file)

assert species_list[i_sens].smiles == 'C=C'

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
        'name': f'corr_truncate_{T}K',        # name for the directory where analysis will be run
        'temperature': T,       # K
        'pressure': ct.one_atm, # Pa
        'composition': X,
        'residence_time': 6.0,  # s
        'volume': 9.5e-5,       # m^3
    })

# Here we reduce the number of samples for much faster runtime, but you'll probably want to do ~100
ezuq.morris_screen.setup_runfiles(working_dir, conditions, N_SAMPLES=8)
# -

# # RUN THE SIMS

for condition in conditions:
    my_settings_file = os.path.join(working_dir, 'morris_screen', condition['name'], 'settings.yaml')
    subprocess.check_call(['python', '-m', 'ezuq.morris_screen', my_settings_file, '0'])

# ## reassemble

ezuq.morris_screen.reassemble_chunks(os.path.join(working_dir, 'morris_screen'))

# # see results

# +
mu_star_threshold = 0.05
results = ezuq.morris_screen.save_reduced_set(
    working_dir,
    conditions,
    i_sens,
    mu_star_threshold=mu_star_threshold,
    verbose=False,
    truncate_in_decomposed_space=True
)

def pad_to_length(input_str, length):
    input_str = str(input_str)
    if len(input_str) > length:
        raise ValueError('Pad length for this string must be at least', len(input_str))
    return input_str + ' ' * (length - len(input_str))

for condition in conditions:
    name = condition['name']

    plt.scatter(results[name]['mu_star'], results[name]['sigma'], s=4)
    plt.axvline(x=np.max(results[name]['mu_star']) * 0.05, color='black', linewidth=0.4)
    plt.xlabel('$\mu *$')
    plt.ylabel('$\sigma$')
    plt.title(condition['name'])
    plt.show()

    # Rank parameter names by mean effect
    contributions = [(x, y) for y, x in sorted(zip(results[name]['mu_star'].data, results[name]['names']))][::-1]
    
    # define the tolerance for considering a parameter to be irrelevant
    threshold = mu_star_threshold * contributions[0][1]
    PAD_LEN = 44
    print('i', '\t', pad_to_length('Parameter', PAD_LEN), 'mu_star')
    print('------------------------------------------------------------------------')
    for i in range(len(contributions)):
    
        if contributions[i][1] < threshold:
            print(f'Reduced to {i} params')
            break
    
        param_name = contributions[i][0]
        print(i, '\t', pad_to_length(param_name, PAD_LEN), f'{contributions[i][1]:0.5f}')

# -


