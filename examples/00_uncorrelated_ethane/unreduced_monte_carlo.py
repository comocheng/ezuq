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

# # Demo of Monte Carlo with no Model Reduction

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
ezuq.monte_carlo.setup_runfiles(working_dir, conditions, morris_dir=None, i_sens=14)
# -

# # RUN THE SIMS (takes a minute or so)

for condition in conditions:
    condition_dir = os.path.join(working_dir, 'monte_carlo', condition['name'])

    my_settings_file = os.path.join(condition_dir, 'settings.yaml')
    print(f'Running sims for {condition["name"]}')
    for i in range(4):
        subprocess.check_call(['python', '-m', 'ezuq.monte_carlo', my_settings_file, str(i)])

# +
# 800K needs some extra runs

condition_dir = os.path.join(working_dir, 'monte_carlo', conditions[0]['name'])
my_settings_file = os.path.join(condition_dir, 'settings.yaml')
for i in range(4, 20):
    subprocess.check_call(['python', '-m', 'ezuq.monte_carlo', my_settings_file, str(i)])
# -

# ## reassemble

for condition in conditions:
    condition_dir = os.path.join(working_dir, 'monte_carlo', condition['name'])
    ezuq.monte_carlo.reassemble_chunks(condition_dir)

# # See results

# ## Check convergence

for z, condition in enumerate(conditions):
    name = condition['name']
    condition_dir = os.path.join(working_dir, 'monte_carlo', name)
    monte_carlo_samples = np.load(os.path.join(condition_dir, 'monte_carlo_results.npy'))
    # remove all zeros
    nonzero_data = monte_carlo_samples[~np.all(monte_carlo_samples == 0, axis=1)]
    
    powers = np.arange(np.log2(int(nonzero_data.shape[0])))
    sample_sizes = np.float_power(2.0, powers)
    sample_sizes = np.array(list(sample_sizes) + [len(nonzero_data) - 1]).astype(int)
    
    i_sp = 14
    
    MEAN_TOL = 0.01
    VAR_TOL = 0.05
    
    # get the convergence sample size
    means = [np.mean(nonzero_data[:x, i_sp]) for x in sample_sizes]
    variances = [np.var(nonzero_data[:x, i_sp]) for x in sample_sizes]
    
    # start here
    N = 1024
    prev_mean = np.mean(nonzero_data[:int(N/2), i_sp])
    converged = False
    while N < nonzero_data.shape[0]:
        current_mean = np.mean(nonzero_data[:N, i_sp])
        if np.abs((current_mean - prev_mean) / prev_mean) < MEAN_TOL:
            print(f'Mean converged at {N} samples')
            break
        N *= 2
        prev_mean = current_mean
    
    N = 1024
    prev_var = np.var(nonzero_data[:int(N/2), i_sp])
    converged = False
    while N < nonzero_data.shape[0]:
        current_var = np.var(nonzero_data[:N, i_sp])
        if np.abs((current_var - prev_var) / prev_var) < VAR_TOL:
            print(f'Variance converged at {N} samples')
            break
        N *= 2
        prev_var = current_var
    
    
    # See what 1% of final value looks like
    total_mean = np.mean(nonzero_data[:, i_sp])
    total_variance = np.var(nonzero_data[:, i_sp])
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    
    # 
    axs[0].set_title(f'Mean Convergence {name}')
    axs[0].plot(sample_sizes, means)
    axs[0].axhline(y=total_mean + MEAN_TOL * total_mean, color='black', linestyle='dashed')
    axs[0].axhline(y=total_mean - MEAN_TOL * total_mean, color='black', linestyle='dashed')
    axs[0].set_xscale('log')
    
    axs[1].set_title(f'Variance Convergence {name}')
    axs[1].plot(sample_sizes, variances)
    axs[1].axhline(y=total_variance + VAR_TOL * total_variance, color='black', linestyle='dashed')
    axs[1].axhline(y=total_variance - VAR_TOL * total_variance, color='black', linestyle='dashed')
    axs[1].set_xscale('log')
    plt.show()

# Get nominal results for plotting
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
    condition_dir = os.path.join(working_dir, 'monte_carlo', name)
    monte_carlo_samples = np.load(os.path.join(condition_dir, 'monte_carlo_results.npy'))
    lower95[z, :] = np.percentile(monte_carlo_samples, 2.5, axis=0)
    upper95[z, :] = np.percentile(monte_carlo_samples, 97.5, axis=0)

# # Plot individual distributions

for z, condition in enumerate(conditions):
    name = condition['name']
    temperature = condition['temperature']
    condition_dir = os.path.join(working_dir, 'monte_carlo', name)
    monte_carlo_samples = np.load(os.path.join(condition_dir, 'monte_carlo_results.npy'))

    # remove all zeros
    nonzero_data = monte_carlo_samples[~np.all(monte_carlo_samples == 0, axis=1)]


    result = plt.hist(nonzero_data[:, i_sp], 48, density=True, alpha=0.6)
    plt.axvline(x=np.mean(nonzero_data[:, i_sp]), color=colors[0], label='Mean')
    plt.axvline(x=np.median(nonzero_data[:, i_sp]), color='black', label='Median')
    
    # Log scale for x axis is a bit confusing
    # bins = np.geomspace(nonzero_data[:, i_sp].min(), nonzero_data[:, i_sp].max(), 64)
    # result = plt.hist(nonzero_data[:, i_sp], bins=bins, density=True, alpha=0.6)
    # plt.axvline(x=np.mean(nonzero_data[:, i_sp]), color=colors[0], label='Mean')
    # plt.axvline(x=np.median(nonzero_data[:, i_sp]), color='black', label='Median')
    # plt.xscale('log')

    plt.title(f'{gas.species_names[i_sp]} PDF - {name}')
    plt.xlabel('Mole Fraction')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# # Plot our final model with errorbars

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
temperatures = [c['temperature'] for c in conditions]
plt.plot(temperatures, nominal_results[:, i_sp], label='Nominal Value')
for z, condition in enumerate(conditions):
    name = condition['name']
    temperature = condition['temperature']
    condition_dir = os.path.join(working_dir, 'monte_carlo', name)
    monte_carlo_samples = np.load(os.path.join(condition_dir, 'monte_carlo_results.npy'))

    # remove all zeros
    nonzero_data = monte_carlo_samples[~np.all(monte_carlo_samples == 0, axis=1)]
    
    # show 95% confidence interval
    label = '_no_label'
    if z == 0:
        label = '95% Confidence Interval'
    plt.fill_between(temperatures, lower95[:, i_sp], nominal_results[:, i_sp], alpha=0.1, color=colors[0], label=label)
    plt.fill_between(temperatures, nominal_results[:, i_sp], upper95[:, i_sp], alpha=0.1, color=colors[0])
    

    plt.boxplot(
        [nonzero_data[:, i_sp]],
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





