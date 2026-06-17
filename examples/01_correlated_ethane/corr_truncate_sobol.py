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

# # Demo of Correlated sobol sampling
#
# ### With model truncation in decomposed space (respects correlations when truncating)

# +
import os
import pickle
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

import SALib.analyze.sobol
# -

working_dir = os.path.abspath('.')
gas = ct.Solution(os.path.join(working_dir, 'chem_annotated.yaml'))

# +
# Specify conditions for the simulation
temperatures = [800, 900, 1000]

i_C2H6 = ezuq.util.get_i_thing_ct({'C': 2, 'H': 6}, gas)
i_C2H4 = ezuq.util.get_i_thing_ct({'C': 2, 'H': 4}, gas)
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
        'output_species_index': i_C2H4,
    })

# Here we reduce the number of samples for much faster runtime, but you'll probably want to do ~1024
ezuq.sobol.setup_runfiles(working_dir, conditions, i_sens=14, N=256)
# -

# # RUN THE SIMS
#
# This can take a while because a 1000-element chunk is really 1000 * (2k+2) samples

for condition in conditions:
    condition_dir = os.path.join(working_dir, 'sobol', condition['name'])

    my_settings_file = os.path.join(condition_dir, 'settings.yaml')
    print(f'Running sims for {condition["name"]}')
    for i in range(1):
        subprocess.check_call(['python', '-m', 'ezuq.sobol', my_settings_file, str(i)])

for condition in conditions:
    condition_dir = os.path.join(working_dir, 'sobol', condition['name'])
    ezuq.sobol.reassemble_chunks(condition_dir)

# # Plot the distribution

# Get nominal results for plotting
i_sp = 14
cantera_file = os.path.join(working_dir, 'chem_annotated.yaml')
gas = ct.Solution(cantera_file)
nominal_results = np.zeros((len(conditions), gas.n_species))
for z, condition in enumerate(conditions):
    nominal_results[z, :] = ezuq.simulation.jsr.run_simulation(gas, condition)

# Get upper/lower 95% confidence intervals for plotting
upper95 = np.zeros((len(conditions), gas.n_species))
lower95 = np.zeros((len(conditions), gas.n_species))
for z, condition in enumerate(conditions):
    name = condition['name']
    condition_dir = os.path.join(working_dir, 'sobol', name)
    sobol_samples = np.concatenate((
        np.load(os.path.join(condition_dir, 'f_y_z.npy')),
        np.load(os.path.join(condition_dir, 'f_y_prime_z_prime.npy'))
    ))
    
    lower95[z, :] = np.percentile(sobol_samples, 2.5, axis=0)
    upper95[z, :] = np.percentile(sobol_samples, 97.5, axis=0)

# +
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for z, condition in enumerate(conditions):
    name = condition['name']
    temperature = condition['temperature']
    condition_dir = os.path.join(working_dir, 'sobol', name)
    sobol_samples = np.concatenate((
        np.load(os.path.join(condition_dir, 'f_y_z.npy')),
        np.load(os.path.join(condition_dir, 'f_y_prime_z_prime.npy'))
    ))

    # remove all zeros
    nonzero_data = sobol_samples[sobol_samples != 0]

    result = plt.hist(nonzero_data, 48, density=True, alpha=0.6)
    plt.axvline(x=np.mean(nonzero_data), color=colors[0], label='Mean')
    plt.axvline(x=np.median(nonzero_data), color='black', label='Median')


    plt.title(f'{gas.species_names[i_sp]} PDF - {name}')
    plt.xlabel('Mole Fraction')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
# -

# # Plot the final model with errorbars

temperatures = [c['temperature'] for c in conditions]
plt.plot(temperatures, nominal_results[:, i_sp], label='Nominal Value')
for z, condition in enumerate(conditions):
    name = condition['name']
    temperature = condition['temperature']
    condition_dir = os.path.join(working_dir, 'sobol', name)
    sobol_samples = np.concatenate((
        np.load(os.path.join(condition_dir, 'f_y_z.npy')),
        np.load(os.path.join(condition_dir, 'f_y_prime_z_prime.npy'))
    ))

    # remove all zeros
    nonzero_data = sobol_samples[sobol_samples != 0]
    
    # show 95% confidence interval
    label = '_no_label'
    if z == 0:
        label = '95% Confidence Interval'
    plt.fill_between(temperatures, lower95[:, i_sp], nominal_results[:, i_sp], alpha=0.1, color=colors[0], label=label)
    plt.fill_between(temperatures, nominal_results[:, i_sp], upper95[:, i_sp], alpha=0.1, color=colors[0])
    

    plt.boxplot(
        [nonzero_data],
        positions=[temperature],
        widths=5,
        showfliers=False,
        # patch_artist=True,
        medianprops=dict(color='black')
    )
plt.title(f'{gas.species_names[i_sp]} Concentration')
plt.xlabel('Temperature (K)')
plt.ylabel('Mole Fraction')
plt.legend()

# # Look at parameter rankings

# +
condition_dir = os.path.join(working_dir, 'sobol', conditions[1]['name'])
problem_desc_file = os.path.join(condition_dir, 'problem_desc.yaml')
with open(problem_desc_file, 'r') as f:
    problem = yaml.load(f, Loader=yaml.FullLoader)

S1, ST = ezuq.sobol.compute_sobol_indices(condition_dir)

# -

S1_results = [(name, x) for x, name in sorted(zip(S1, problem['names']))][::-1]
for i in range(len(S1_results)):
    print(i, S1_results[i][0])

ST_results = [(name, x) for x, name in sorted(zip(ST, problem['names']))][::-1]
for i in range(len(ST_results)):
    print(i, ST_results[i][0])


