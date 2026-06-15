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

# # Demo of Morris Screening

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
        'name': f'{T}K',        # name for the directory where analysis will be run
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

morris_dir = os.path.join(working_dir, 'morris_screen')
morris_samples = np.load(os.path.join(morris_dir, 'morris_samples.npy'))
with open(os.path.join(morris_dir, 'problem_desc.yaml'), 'rb') as f:
    problem = yaml.load(f, Loader=yaml.FullLoader)

for condition in conditions:
    morris_y = np.load(os.path.join(morris_dir, condition['name'], 'morris_sim_results.npy'))

    nominal_values = ezuq.simulation.jsr.run_simulation(gas, condition)
    for i in range(morris_y.shape[0]):
        if np.isnan(morris_y[i, :]).any():
            morris_y[i, :] = nominal_values


    # Do analysis in physical parameter space
    # load the covariance matrices
    thermo_covariance_matrix = np.load(os.path.join(working_dir, 'thermo_covariance_matrix.npy'))
    kinetic_covariance_matrix = np.load(os.path.join(working_dir, 'kinetic_covariance_matrix.npy'))
    
    z = scipy.stats.norm.ppf(morris_samples)
    z_thermo = z[:, :gas.n_species]
    z_kinetic = z[:, gas.n_species:]
    
    
    L_thermo = np.linalg.cholesky(thermo_covariance_matrix)
    assert np.isclose(L_thermo @ L_thermo.T, thermo_covariance_matrix).all()
    thermo_perturbations = (L_thermo @ z_thermo.T).T * 4184  # convert RMG-UQ's kcal/mol to J/mol
    
    
    L_kinetic = np.linalg.cholesky(kinetic_covariance_matrix)
    assert np.isclose(L_kinetic @ L_kinetic.T, kinetic_covariance_matrix).all()
    kinetic_perturbations = (L_kinetic @ z_kinetic.T).T  # in lnk space


    w_thermo = (L_thermo @ z_thermo.T).T
    w_kinetic = (L_kinetic @ z_kinetic.T).T
    w = np.concatenate((w_thermo, w_kinetic), axis=1)

    
    physical_result = SALib.analyze.morris.analyze(problem, w, morris_y[:, i_sens], scaled=True)
    # independent_result = SALib.analyze.morris.analyze(problem, z, morris_y[:, i_sens], scaled=False)

    plt.scatter(physical_result['mu_star'], physical_result['sigma'], s=4)
    plt.axvline(x=np.max(physical_result['mu_star']) * 0.05, color='black', linewidth=0.4)
    plt.xlabel('$\mu *$')
    plt.ylabel('$\sigma$')
    plt.title(condition['name'])
    plt.show()


    # Rank parameter names by mean effect
    contributions = [(x, y) for y, x in sorted(zip(physical_result['mu_star'].data, problem['names']))][::-1]
    
    # define the tolerance for considering a parameter to be irrelevant
    threshold = 0.05 * contributions[0][1]  # use 0.01 for tighter tolerance
    k_params = []
    g_params = []
    
    species_names = [x.to_chemkin() for x in species_list]
    reaction_names = [x.to_chemkin(species_list, kinetics=False) for x in reaction_list]
    for i in range(len(contributions)):
    
        if contributions[i][1] < threshold:
            print(f'Reduced to {i} params')
            break
    
        name = contributions[i][0]
        print(i, name)
        if name in species_names:
            g_params.append(species_names.index(name))
        elif name in reaction_names:
            k_params.append(reaction_names.index(name))
        else:
            raise ValueError(f'could not identify parameter with name {name}')


    # save the problem description for this condition. Different conditions will have difference reduced parameter sets
    morris_screen_result = {
        'g_params': g_params,
        'k_params': k_params,
        'num_vars': len(g_params) + len(k_params),
        'g_param_names': [species_list[i].to_chemkin() for i in g_params],
        'k_param_names': [reaction_list[i].to_chemkin(species_list, kinetics=False) for i in k_params],
    }
    
    with open(os.path.join(morris_dir, condition['name'], 'morris_screen_set.yaml'), 'w') as f:
        yaml.dump(morris_screen_result, f, default_flow_style=False)


