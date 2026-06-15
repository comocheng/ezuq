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

# +
import os
import csv
import pickle
import cantera as ct
import numpy as np
import rmgpy.chemkin
import rmgpy.tools.plot

import ezuq.util
import ezuq.simulation.jsr

import matplotlib.pyplot as plt
# %matplotlib inline
# -

# # Run sensitivity

# +
mech_yaml = 'chem_annotated.yaml'
gas = ct.Solution(mech_yaml)

temperatures = [800, 900, 1000]

sensitivity = np.zeros((len(temperatures), gas.n_reactions + gas.n_species, gas.n_species))
mole_fracs = np.zeros((len(temperatures), gas.n_species))
mass_fracs = np.zeros((len(temperatures), gas.n_species))

# Not a real experiment
i_C2H6 = ezuq.util.get_i_thing_ct({'C': 2, 'H': 6}, gas)
i_O2 = ezuq.util.get_i_thing_ct({'O': 2}, gas)
i_Ar = ezuq.util.get_i_thing_ct({'Ar': 1}, gas)  # He for Nancy data
x_C2H6 = 0.18
x_O2 = 0.64
x_Ar = 1.0 - x_C2H6 - x_O2
X = f'{gas.species_names[i_C2H6]}: {x_C2H6}, {gas.species_names[i_O2]}: {x_O2}, {gas.species_names[i_Ar]}: {x_Ar}'

for i, T in enumerate(temperatures):
    settings = {
        'temperature': T,       # K
        'pressure': ct.one_atm, # Pa
        'composition': X,
        'residence_time': 6.0,  # s
        'volume': 9.5e-5,       # m^3
    }
    mole_frac_i, mass_frac_i, sensitivity_i = ezuq.simulation.jsr.run_simulation_with_sensitivity(gas, settings)

    mole_fracs[i, :] = mole_frac_i
    mass_fracs[i, :] = mass_frac_i
    sensitivity[i, :] = sensitivity_i
    
# -

# ## Plot top concentrations

for i in range(gas.n_species):
    if mole_fracs[2, i] > 1e-3:
        plt.plot(temperatures, mole_fracs[:, i], label=gas.species_names[i])
plt.legend()
plt.ylabel('Mole Fraction')
plt.xlabel('Temperature (K)')

# ## Plot top sensitivities

# +
i_sp = ezuq.util.get_i_thing_ct({'C': 2, 'H': 4}, gas)

top_N = 5
sens_sp = i_sp

max_conc = np.max(np.abs(sensitivity[:, :, sens_sp]), axis=0)
indices = np.arange(sensitivity.shape[1])
sorted_order = [x for _, x in sorted(zip(max_conc, indices))][::-1]
for i in range(top_N):
    j = sorted_order[i]
    if j > gas.n_reactions:
        label = gas.species_names[j - gas.n_reactions]
    else:
        label = gas.reaction_equations()[j]
    plt.plot(temperatures, sensitivity[:, j, sens_sp], label=label)

plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel('Sensitivity')
# -

# ## Convert from Cantera's mass-based sensitivity to RMG's mole-based sensitivity
# This can take a minute or two if the model's large or there are lots of temperatures

# +
molecular_weights = gas.molecular_weights / 1000.0  # convert from kg/kmol to kg/mol
mol_sensitivities = np.zeros_like(sensitivity) + np.nan

for i in range(sensitivity.shape[1]):  # repeat over perturbed parameter p
    for k in range(sensitivity.shape[2]):  # repeat over sensitive species concentration y_k
        
        sum_xj_Wj = np.zeros(sensitivity.shape[0])
        for j in range(gas.n_species):
            sum_xj_Wj += mass_fracs[:, j] / molecular_weights[j]

        low_d_high = np.multiply(np.multiply(sum_xj_Wj, mass_fracs[:, k]), sensitivity[:, i, k] / molecular_weights[k])

        sum_xj_Winv_dx_dp = np.zeros(sensitivity.shape[0])
        for j in range(gas.n_species):
            sum_xj_Winv_dx_dp += np.multiply(mass_fracs[:, j], sensitivity[:, i, j] / molecular_weights[j])
        high_d_low = np.multiply(sum_xj_Winv_dx_dp, mass_fracs[:, k] / molecular_weights[k])

        mol_sensitivities[:, i, k] = np.divide(np.divide((low_d_high - high_d_low), np.float_power(sum_xj_Wj, 2.0)), mole_fracs[:, k])
# -

# ## Plot again
# May have changed things slightly

# +
max_conc = np.max(np.abs(mol_sensitivities[:, :, sens_sp]), axis=0)
indices = np.arange(mol_sensitivities.shape[1])
sorted_order = [x for _, x in sorted(zip(max_conc, indices))][::-1]
for i in range(top_N):
    j = sorted_order[i]
    if j > gas.n_reactions:
        label = gas.species_names[j - gas.n_reactions]
    else:
        label = gas.reaction_equations()[j]
    plt.plot(temperatures, mol_sensitivities[:, j, sens_sp], label=label)

plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel('Sensitivity')
# -

# ## Save in format RMG-UQ expects

# +
species_list, reaction_list = rmgpy.chemkin.load_chemkin_file('chem_annotated.inp', 'species_dictionary.txt')

with open('ct2rmg_rxn.pickle', 'rb') as f:
    ct2rmg_rxn = pickle.load(f)

os.makedirs('solver', exist_ok=True)
# make a sensitivity worksheet for each species
sens_worksheet = []
for spec in species_list:
    csvfile_path = os.path.join('solver',
                                'sensitivity_{0}_SPC_{1}.csv'.format(1, spec.index))
    sens_worksheet.append(csvfile_path)
# Write sensitivities to CSV files, one file per sensitive species
sensitivity_threshold = 1e-12
for j in range(len(species_list)):
    with open(sens_worksheet[j], 'w') as outfile:
        # species order will be the same between RMG and Cantera, so this is dine
        species_name = species_list[j].to_chemkin()

        
        headers = ['Time (s)']

        worksheet = csv.writer(outfile)
        reactions_above_threshold = []  # includes species too

        
        for i in range(gas.n_reactions + gas.n_species):
            for t in range(mol_sensitivities.shape[0]):  # loop over time steps
                if abs(mol_sensitivities[t, i, j]) > sensitivity_threshold:
                    reactions_above_threshold.append(i)
                    break

        # need conversion from Cantera to RMG and back

        col_names = []
        for i in reactions_above_threshold:
            if i < gas.n_reactions:
                i_rmg = ct2rmg_rxn[i]
                col_names.append(f'dln[{species_name}]/dln[k{i_rmg + 1}]: {reaction_list[i_rmg].to_chemkin(species_list, kinetics=False)}')
                # print(gas.reaction_equations()[i])
                # print(reaction_list[i_rmg].to_chemkin(species_list, kinetics=False))
                # print()    
                # i is a cantera reaction that needs mapping back to RMG
            else:
                col_names.append(f'dln[{species_name}]/dG[{species_list[i - gas.n_reactions].to_chemkin()}]')
        headers.extend(col_names)
        worksheet.writerow(headers)
        for t in range(len(temperatures)):
            row = [temperatures[t]]
            row.extend([mol_sensitivities[t][i, j] for i in reactions_above_threshold])
            worksheet.writerow(row)
# -


