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

# # Demo Notebook for Sobol Sampling with Model Reduction

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

# +
# ezuq.sobol.CONFIDENCE_INTERVAL = 0.99
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
        'name': f'{T}K',        # name for the directory where analysis will be run
        'temperature': T,       # K
        'pressure': ct.one_atm, # Pa
        'composition': X,
        'residence_time': 6.0,  # s
        'volume': 9.5e-5,       # m^3
    })

# Here we reduce the number of samples for much faster runtime, but you'll probably want to do ~1024
ezuq.sobol.setup_runfiles(working_dir, conditions, i_sens=14, N=256)
# -

# # RUN THE SIMS

for condition in conditions:
    condition_dir = os.path.join(working_dir, 'sobol', condition['name'])

    my_settings_file = os.path.join(condition_dir, 'settings.yaml')
    print(f'Running sims for {condition["name"]}')
    for i in range(6):
        
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
    sobol_samples = np.load(os.path.join(condition_dir, 'sobol_results.npy'))
    lower95[z, :] = np.nanpercentile(sobol_samples, 2.5, axis=0)
    upper95[z, :] = np.nanpercentile(sobol_samples, 97.5, axis=0)

# +
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for z, condition in enumerate(conditions):
    name = condition['name']
    temperature = condition['temperature']
    condition_dir = os.path.join(working_dir, 'sobol', name)
    sobol_samples = np.load(os.path.join(condition_dir, 'sobol_results.npy'))

    # set all zero_data and nans to nominal values
    for i in range(sobol_samples.shape[0]):
        if np.all(sobol_samples[i, :] == 0):
            sobol_samples[i, :] = nominal_results[z, :]
        elif np.isnan(sobol_samples[i, i_sp]):
            sobol_samples[i, :] = nominal_results[z, :]

    result = plt.hist(sobol_samples[:, i_sp], 48, density=True, alpha=0.6)
    plt.axvline(x=np.mean(sobol_samples[:, i_sp]), color=colors[0], label='Mean')
    plt.axvline(x=np.median(sobol_samples[:, i_sp]), color='black', label='Median')
    plt.axvline(x=nominal_results[z, i_sp], color='red', linestyle='dashed', label='Nominal Value')


    plt.title(f'{gas.species_names[i_sp]} PDF - {name}')
    plt.xlabel('Mole Fraction')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
# -



# # Plot the final model with errorbars

# +

temperatures = [c['temperature'] for c in conditions]
plt.plot(temperatures, nominal_results[:, i_sp], label='Nominal Value')
for z, condition in enumerate(conditions):
    name = condition['name']
    temperature = condition['temperature']
    condition_dir = os.path.join(working_dir, 'sobol', name)
    sobol_samples = np.load(os.path.join(condition_dir, 'sobol_results.npy'))

    # set all zero_data and nans to nominal values
    for i in range(sobol_samples.shape[0]):
        if np.all(sobol_samples[i, :] == 0):
            sobol_samples[i, :] = nominal_results[z, :]
        elif np.isnan(sobol_samples[i, i_sp]):
            sobol_samples[i, :] = nominal_results[z, :]
    
    # show 95% confidence interval
    label = '_no_label'
    if z == 0:
        label = '95% Confidence Interval'
    plt.fill_between(temperatures, lower95[:, i_sp], nominal_results[:, i_sp], alpha=0.1, color=colors[0], label=label)
    plt.fill_between(temperatures, nominal_results[:, i_sp], upper95[:, i_sp], alpha=0.1, color=colors[0])
    

    plt.boxplot(
        [sobol_samples[:, i_sp]],
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
# -

print(lower95)

# # Look at some rankings

# +
# Load the Sobol X
sobol_X = np.load(os.path.join(working_dir, 'sobol', conditions[2]['name'], 'sobol_samples.npy'))
with open(os.path.join(working_dir, 'sobol', conditions[2]['name'], 'problem_desc.yaml'), 'rb') as f:
    sobol_problem = yaml.load(f, Loader=yaml.FullLoader)

sobol_y = np.load(os.path.join(working_dir, 'sobol', conditions[2]['name'], 'sobol_results.npy'))
# set all zero_data and nans to nominal values
for i in range(sobol_y.shape[0]):
    if np.all(sobol_y[i, :] == 0):
        sobol_y[i, :] = nominal_results[z, :]
    elif np.isnan(sobol_y[i, i_sp]):
        sobol_y[i, :] = nominal_results[z, :]

problem_desc_file = os.path.join(working_dir, 'sobol', conditions[2]['name'], 'problem_desc.yaml')
with open(problem_desc_file, 'r') as f:
    problem = yaml.load(f, Loader=yaml.FullLoader)

# -



sobol_problem = {
    'num_vars': problem['num_vars'],
    'bounds': problem['bounds'],
    'names': problem['names'],
}

Si = SALib.analyze.sobol.analyze(sobol_problem, sobol_y[:, i_sp], calc_second_order=False, seed=400)

# +
# Show S1 rankings

S1_results = [(name, x) for x, name in sorted(zip(Si['S1'], sobol_problem['names']))][::-1]
for i in range(len(S1_results)):
    print(i, S1_results[i][0], S1_results[i][1])
    # print(i, S1_results[i][0])

# +
# Show S1 rankings

ST_results = [(name, x) for x, name in sorted(zip(Si['ST'], sobol_problem['names']))][::-1]
for i in range(len(ST_results)):
    print(i, ST_results[i][0], ST_results[i][1])

# +
# show the covariance matrix used

thermo_covariance_matrix = np.load(os.path.join(working_dir, 'thermo_covariance_matrix.npy'))
thermo_covariance_matrix_subset = thermo_covariance_matrix[np.ix_(problem['g_params'], problem['g_params'])]


thermo_covariance_matrix_reduced = np.zeros_like(thermo_covariance_matrix)
for i in range(thermo_covariance_matrix_reduced.shape[0]):
    if i in problem['g_params']:
        thermo_covariance_matrix_reduced[i, i] = thermo_covariance_matrix[i, i]


thermo_covariance_matrix_reduced[np.ix_(problem['g_params'], problem['g_params'])] = thermo_covariance_matrix_subset
plt.matshow(thermo_covariance_matrix_subset)
plt.matshow(thermo_covariance_matrix_reduced)
# -

# # look at the shape of the distributions sampled

# +
thermo_covariance_matrix = np.load(os.path.join(working_dir, 'thermo_covariance_matrix.npy'))
g_params = problem['g_params']
thermo_uniform_perturbations = sobol_X[:, :len(g_params)]
L_thermo = np.linalg.cholesky(thermo_covariance_matrix)
assert np.isclose(L_thermo @ L_thermo.T, thermo_covariance_matrix).all()
z_thermo_reduced = scipy.stats.norm.ppf(thermo_uniform_perturbations)  # transform the unit uniforms to standard normals
z_thermo = np.zeros((z_thermo_reduced.shape[0], gas.n_species))

for i, sp_index in enumerate(g_params):
    z_thermo[:, sp_index] = z_thermo_reduced[:, i]

thermo_perturbations = (L_thermo @ z_thermo.T).T * 4184  # convert RMG-UQ's kcal/mol to J/mol

# -

for i in range(gas.n_species):
    if np.sum(thermo_perturbations[:, i]) > 0:
        plt.hist(thermo_perturbations[:, i], 32, density=True, alpha=0.2)
        

plt.hist(thermo_perturbations[:, 4], 32)



L_thermo.shape

thermo_covariance_matrix.shape

gas.n_species


