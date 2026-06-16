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
import glob
import shutil
import yaml
import csv
import cantera as ct
import pandas as pd
import numpy as np
import pickle
import rmgpy.chemkin
import rmgpy.tools.plot

import rmgpy.tools.uncertainty
import matplotlib.pyplot as plt
# %matplotlib inline

starting_dir = os.path.abspath('.')
# -

# # 1. Load model and database

# +
chemkin_file = 'chem_annotated.inp'
dict_file = 'species_dictionary.txt'
cantera_file = 'chem_annotated.yaml'

with open('ct2rmg_rxn.pickle', 'rb') as f:
    ct2rmg_rxn = pickle.load(f)

try:
    REPO_DIR = os.environ['UQ_REFORMULATION_REPO']
except KeyError:
    REPO_DIR = '/scratch/harris.se/guassian_scratch/uq_reformulation_paper/'


family_dirs = glob.glob(os.path.join(REPO_DIR, '03_BMKinetics_Covariances', 'yes_rank_calc', '*'))
family_dirs = sorted([x for x in family_dirs if os.path.isdir(x)])
display(family_dirs)


# Initialize the Uncertainty class instance and load the model
# sensitivities have been copied over to the 04_Local_Uncertainty folder so plots generated don't crowd the 02_JSR_sensitivities folder
uncertainty = rmgpy.tools.uncertainty.Uncertainty(
    output_directory='../ethane/',
    kinetic_covariance_rules=family_dirs,
)
uncertainty.load_model(chemkin_file, dict_file)

# -

uncertainty.load_database(
    thermo_libraries=[
        'primaryThermoLibrary',
    ],
    kinetics_families='default',
    reaction_libraries=[
        'BurkeH2O2inArHe',
    ],
)

uncertainty.extract_sources_from_model()
uncertainty.assign_parameter_uncertainties(correlated=True, use_rank=True)


# # Pick a sensitive species and do local analysis

# +
def get_i_thing(thing, thing_list):
    for i in range(len(thing_list)):
        if thing.is_isomorphic(thing_list[i]):
            return i
    assert False

ref_sp = rmgpy.species.Species(smiles='C=C')
sensitive_species = [uncertainty.species_list[get_i_thing(ref_sp, uncertainty.species_list)]]

temperatures = np.load('../ethane/temperatures.npy')
concentrations = np.load('../ethane/mole_fracs.npy')
# -

# # get and save the covariance matrices

# +
thermo_covariance_matrix = uncertainty.get_thermo_covariance_matrix()
kinetic_covariance_matrix = uncertainty.get_kinetic_covariance_matrix()

np.save('thermo_covariance_matrix.npy', thermo_covariance_matrix)
np.save('kinetic_covariance_matrix.npy', kinetic_covariance_matrix)
# -

variances = uncertainty.get_variance_across_x(sensitive_species[0], correlated=True)

# # plot the species concentration with and without error bars

# ### No error bars on model for better scaling

# +
i_sp = get_i_thing(sensitive_species[0], uncertainty.species_list)

plt.plot(temperatures, concentrations[:, i_sp], label='RMG Sim')
plt.xlabel('Temperature (K)')
plt.ylabel('Mole Fraction')
plt.legend()
plt.title(sensitive_species[0].to_chemkin())
ax = plt.gca()
base_ylim = ax.get_ylim()

# plt.yscale('log')
plt.show()

# -

# ### Plot with local uncertainty error bars on model

# +

plt.plot(temperatures, concentrations[:, i_sp], label='RMG Sim')
plt.xlabel('Temperature (K)')
plt.ylabel('Mole Fraction')
plt.legend()
plt.title(sensitive_species[0].to_chemkin())

sigmas = np.sqrt(variances)

# Drawing 95th percent confidence interval
upper_bound = np.exp(1.96 * sigmas) * concentrations[:, i_sp]
lower_bound = concentrations[:, i_sp] / np.exp(sigmas * 1.96)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.fill_between(temperatures, concentrations[:, i_sp], upper_bound, alpha=0.5, color=colors[0])
plt.fill_between(temperatures, lower_bound, concentrations[:, i_sp], alpha=0.5, color=colors[0])
plt.title(sensitive_species[0].to_chemkin())
# plt.ylim(base_ylim)
plt.yscale('log')

plt.show()
# -
plt.matshow(kinetic_covariance_matrix)





# ### Describe top contributors at a specific temperature

# +
temperature = 800
output = uncertainty.local_analysis(sensitive_species=sensitive_species, correlated=True, t=temperature)
var, reaction_delta, species_delta = output[sensitive_species[0]]

print(rmgpy.tools.uncertainty.process_local_results(output, sensitive_species, number=10)[1])

# # copy those plots into this local folder
r_path = os.path.join(uncertainty.output_directory, 'uncorrelated', f'kineticsLocalUncertainty_{sensitive_species[0].to_chemkin()}.png')
t_path = os.path.join(uncertainty.output_directory, 'uncorrelated', f'thermoLocalUncertainty_{sensitive_species[0].to_chemkin()}.png')
shutil.copyfile(r_path, os.path.join(starting_dir, os.path.basename(r_path)))
shutil.copyfile(t_path, os.path.join(starting_dir, os.path.basename(t_path)))
# -
# ### Rank the parameters without separating thermo/kinetics

param_name1 = [x[0] for x in reaction_delta + species_delta]
param_name2 = [x[1] for x in reaction_delta + species_delta]
param_var = [x[2] for x in reaction_delta + species_delta]

var, reaction_delta, species_delta = output[sensitive_species[0]]

sorted_result = [(x, y, z) for x, y, z in sorted(zip(param_var, param_name1, param_name2))][::-1]
for i in range(20):
    # print(i, sorted_result[i][2], sorted_result[i][0])
    print(i, sorted_result[i][2])

# +
contributors = [entry[1] for entry in reaction_delta] + [entry[1] for entry in species_delta]
contributions = [entry[2] for entry in reaction_delta] + [entry[2] for entry in species_delta]

ranked_contributions = [(name, x) for x, name in sorted(zip(contributions, contributors))][::-1]

threshold = 0.99  # percent of local uncertainty to analyze for global uncertainty
N_include = 0
for N_include in range(len(ranked_contributions)):
    if np.sum([x[1] for x in ranked_contributions[:N_include]]) / var > threshold:
        break
# display(ranked_contributions[:N_include])

print(N_include)



# -
ranked_contributions[:10]

# # Save top 10 in a problem_desc.yaml for MC run

# +
monte_carlo_condition_dir = '/scratch/harris.se/guassian_scratch/uq_reformulation_paper_runs/01_uncorrelated_no_rank_no_BM_cov/local_top_10_mc/600K'

k_params = []
g_params = []

species_names = [x.to_chemkin() for x in uncertainty.species_list]
reaction_names = [x.to_chemkin(uncertainty.species_list, kinetics=False) for x in uncertainty.reaction_list]
for i in range(10):
    name = ranked_contributions[i][0]
    if name in species_names:
        g_params.append(species_names.index(name))
    elif name in reaction_names:
        k_params.append(reaction_names.index(name))
    else:
        raise ValueError(f'could not identify parameter with name {name}')


# save the problem description for this condition. Different conditions will have difference reduced parameter sets
problem = {
    'g_params': g_params,
    'k_params': k_params,
    'num_vars': len(g_params) + len(k_params),
    'g_param_names': [uncertainty.species_list[i].to_chemkin() for i in g_params],
    'k_param_names': [uncertainty.reaction_list[i].to_chemkin(uncertainty.species_list, kinetics=False) for i in k_params],
}

with open(os.path.join(monte_carlo_condition_dir, 'problem_desc.yaml'), 'w') as f:
    yaml.dump(problem, f, default_flow_style=False)
# -

reaction_names = [rxn.to_chemkin(uncertainty.species_list, kinetics=False) for rxn in uncertainty.reaction_list]
species_names = [sp.to_chemkin() for sp in uncertainty.species_list]

for i in range(N_include):
    # name = sorted_result[i][2]
    # if '<=>' in name:
    #     display(uncertainty.reaction_list[reaction_names.index(name)])
    # else:
    #     display(uncertainty.species_list[species_names.index(name)])
    print(i, sorted_result[i][2], sorted_result[i][0])

display(uncertainty.species_list[44])

# +
# confirm the sensitivities???
# -

csvfile_path = '/scratch/harris.se/guassian_scratch/uq_reformulation_paper/02_JSR_sensitivities/solver/sensitivity_1_SPC_11.csv'

T_K, data_list = rmgpy.tools.plot.parse_csv_data(csvfile_path)

list(T_K.data).index(600)

T_K.data[30]

col_names = [data.label for data in data_list]
sens_values = [data.data[30] for data in data_list]

# +
# rank sensitivities


sens_results = [(y, x) for x, y in sorted(zip(sens_values, col_names))][::-1]
for i in range(10):
    print(i, sens_results[i][0], sens_results[i][1])
# -


