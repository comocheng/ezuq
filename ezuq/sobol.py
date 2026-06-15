"""A module for running Monte Carlo samples of reaction model"""


import os
import re
import glob
import sys
import pickle
import shutil
import numpy as np
import yaml
import cantera as ct
import rmgpy.chemkin
import scipy.stats

import SALib.analyze.sobol
import SALib.sample.sobol

import ezuq.util
from ezuq.simulation.jsr import run_simulation

CHUNK_SIZE = 1000

def setup_runfiles(working_dir, conditions, morris_dir='?', i_sens=None, N=1024, SEED=400):
    """Set up the runfiles for Sobol Sampling
    working_dir should be the directory where the RMG and Cantera mechanisms are saved.

    optional morris_dir to use the results of screening
    if no morris_dir is provided and the string '?' is provided, it will search for it in the usual location.
    If morrid_dir is None, this will set up the runfiles for a full Monte Carlo sampling, which is probably too expensive to run to convergence

    i_sens is the index of the output variable to use for screening if using morris_dir
    """

    sobol_dir = os.path.join(working_dir, 'sobol')
    os.makedirs(sobol_dir, exist_ok=True)

    if morris_dir == '?':
        # try to find it
        if os.path.exists(os.path.join(working_dir, 'morris_screen')) and \
                os.path.exists(os.path.join(working_dir, 'morris_screen', 'morris_samples.npy')) and \
                os.path.exists(os.path.join(working_dir, 'morris_screen', 'problem_desc.yaml')):
            morris_dir = os.path.join(working_dir, 'morris_screen')

    # load the covariance matrices
    thermo_covariance_matrix = np.load(os.path.join(working_dir, 'thermo_covariance_matrix.npy'))
    kinetic_covariance_matrix = np.load(os.path.join(working_dir, 'kinetic_covariance_matrix.npy'))


    # confirm this matches the RMG mechanism
    chemkin_file = os.path.join(working_dir, 'chem_annotated.inp')
    dictionary_file = os.path.join(working_dir, 'species_dictionary.txt')
    species_list, reaction_list = rmgpy.chemkin.load_chemkin_file(chemkin_file, dictionary_file)
    assert len(species_list) == thermo_covariance_matrix.shape[0], "Thermo covariance matrix size does not match number of species"
    assert len(reaction_list) == kinetic_covariance_matrix.shape[0], "Kinetic covariance matrix size does not match number of reactions"

    cantera_file = os.path.join(working_dir, 'chem_annotated.yaml')
    gas = ct.Solution(cantera_file)
    with open(os.path.join(working_dir, 'ct2rmg_rxn.pickle'), 'rb') as f:
        ct2rmg_rxn = pickle.load(f)
    
    assert gas.n_species == thermo_covariance_matrix.shape[0], "Thermo covariance matrix size does not match number of species in Cantera mechanism"
    assert gas.n_reactions == len(ct2rmg_rxn), "Kinetic covariance matrix size does not match number of reactions in Cantera mechanism"
    assert len(set(ct2rmg_rxn.values())) == len(reaction_list), "Reactions in Cantera mechanism do not match reactions in RMG mechanism"

    # make the condition dirs, use the input conditions to name things
    ezuq.util.setup_condition_dirs(sobol_dir, conditions)
    if morris_dir and i_sens is not None:
        raise NotImplementedError('Sobol using model reduction not implemented yet')
    else:
        print('No reduced model params provided, this will run sampling on the full set')

        # Define the problem using SALib format
        # we need to clip the bounds to avoid infinity in the transformation to normal space.
        confidence_interval = 0.95
        alpha = (1 - confidence_interval) / 2

        problem = {
            'num_vars': len(species_list) + len(reaction_list),
            'names': [sp.to_chemkin() for sp in species_list] + 
                    [rxn.to_chemkin(species_list, kinetics=False) for rxn in reaction_list],
            'bounds': [[alpha, 1 - alpha]] * (len(species_list) + len(reaction_list)),  # (slightly clipped) unit uniforms, we'll handle the actual translation to valid perturbations later on
        }
        with open(os.path.join(sobol_dir, 'problem_desc.yaml'), 'w') as f:
            yaml.dump(problem, f)

        # Generate Sobol samples (takes a minute)
        X = SALib.sample.sobol.sample(problem, N=N, calc_second_order=False, seed=SEED)
        print(f'Generated {X.shape[0]} samples with {X.shape[1]} variables')
        np.save(os.path.join(sobol_dir, 'sobol_samples.npy'), X)



    # if no Morris dir is provided, we'll run a full Monte Carlo sampling, which requires no problem description since all parameters are varied

    # copy the slurm script into the Sobol dir
    shutil.copyfile(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'SLURM', 'run_sobol.sh'), os.path.join(sobol_dir, 'run_sobol.sh'))


def run_chunk(settings_yaml, chunk_index):
    """Run a chunk of the sobol simulations
    Assumes the following directory structure:
    working_dir/
        chem_annotated.inp
        species_dictionary.txt
        chem_annotated.yaml
        ct2rmg_rxn.pickle
        thermo_covariance_matrix.npy
        kinetic_covariance_matrix.npy
        sobol/
            problem_desc.yaml
            550K/
                settings.yaml
                sobol_y/
            650K/
                settings.yaml
                sobol_y/
            750K/
                settings.yaml
                sobol_y/
    """

    condition_dir = os.path.dirname(os.path.abspath(settings_yaml))
    sobol_dir = os.path.dirname(condition_dir)
    working_dir = os.path.dirname(sobol_dir)
    results_dir = os.path.join(condition_dir, 'sobol_y')
    output_filename = os.path.join(results_dir, f'y_{chunk_index:04}.npy')
    os.makedirs(results_dir, exist_ok=True)

    # Load relevant files and check for consistency
    with open(settings_yaml, 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    cantera_file = os.path.join(working_dir, 'chem_annotated.yaml')
    gas = ct.Solution(cantera_file)
    chemkin_file = os.path.join(working_dir, 'chem_annotated.inp')
    dict_file = os.path.join(working_dir, 'species_dictionary.txt')
    species_list, reaction_list = rmgpy.chemkin.load_chemkin_file(chemkin_file, dict_file)
    with open(os.path.join(working_dir, 'ct2rmg_rxn.pickle'), 'rb') as f:
        ct2rmg_rxn = pickle.load(f)
    ct2rmg_matrix = np.zeros((gas.n_reactions, len(reaction_list)))
    for ct_index, rmg_index in ct2rmg_rxn.items():
        ct2rmg_matrix[ct_index, rmg_index] = 1

    thermo_covariance_matrix = np.load(os.path.join(working_dir, 'thermo_covariance_matrix.npy'))
    kinetic_covariance_matrix = np.load(os.path.join(working_dir, 'kinetic_covariance_matrix.npy'))
    assert len(species_list) == thermo_covariance_matrix.shape[0], "Thermo covariance matrix size does not match number of species"
    assert len(reaction_list) == kinetic_covariance_matrix.shape[0], "Kinetic covariance matrix size does not match number of reactions"
    assert gas.n_species == thermo_covariance_matrix.shape[0], "Thermo covariance matrix size does not match number of species in Cantera mechanism"
    assert gas.n_reactions == len(ct2rmg_rxn), "Kinetic covariance matrix size does not match number of reactions in Cantera mechanism"
    assert len(set(ct2rmg_rxn.values())) == len(reaction_list), "Reactions in Cantera mechanism do not match reactions in RMG mechanism"

    y = np.zeros((CHUNK_SIZE, gas.n_species))

    rng = np.random.default_rng(chunk_index)

    # save copies of all thermo for faster perturbation
    thermo_copies = []
    for sp_index in range(gas.n_species):
        thermo_copies.append(ct.Species().from_dict(gas.species()[sp_index].input_data.copy()))

    problem_desc_file = os.path.join(condition_dir, 'problem_desc.yaml')
    if not os.path.exists(problem_desc_file):
        # no problem dscription with parameter reduction was done, so this will be a full Monte Carlo sampling
        g_params = list(range(gas.n_species))
        k_params = list(range(len(reaction_list)))

        thermo_perturbations = rng.multivariate_normal(mean=np.zeros(thermo_covariance_matrix.shape[0]), cov=thermo_covariance_matrix, size=CHUNK_SIZE) * 4184  # convert to J/mol
        kinetic_perturbations = rng.multivariate_normal(mean=np.zeros(kinetic_covariance_matrix.shape[0]), cov=kinetic_covariance_matrix, size=CHUNK_SIZE)

    else:
        with open(problem_desc_file, 'r') as f:
            problem = yaml.load(f, Loader=yaml.FullLoader)

        g_params = problem['g_params']
        k_params = problem['k_params']

        # This is only okay if there are no off-diagonals
        # count the off_diagonals
        if not ezuq.util.is_diagonal(thermo_covariance_matrix) or not ezuq.util.is_diagonal(kinetic_covariance_matrix):
            raise NotImplementedError("Parameter reduction with non-diagonal covariance matrices is not implemented yet.")

        # reduce the thermo covariance matrix to the species in g_params
        thermo_covariance_matrix_subset = thermo_covariance_matrix[np.ix_(g_params, g_params)]
        kinetic_covariance_matrix_subset = kinetic_covariance_matrix[np.ix_(k_params, k_params)]

        thermo_perturbations_subset = rng.multivariate_normal(mean=np.zeros(thermo_covariance_matrix_subset.shape[0]), cov=thermo_covariance_matrix_subset, size=CHUNK_SIZE) * 4184  # convert to J/mol
        kinetic_perturbations_subset = rng.multivariate_normal(mean=np.zeros(kinetic_covariance_matrix_subset.shape[0]), cov=kinetic_covariance_matrix_subset, size=CHUNK_SIZE)

        thermo_perturbations = np.zeros((CHUNK_SIZE, gas.n_species))
        kinetic_perturbations = np.zeros((CHUNK_SIZE, len(reaction_list)))

        for i, g_param in enumerate(g_params):
            thermo_perturbations[:, g_param] = thermo_perturbations_subset[:, i]
        for j, k_param in enumerate(k_params):
            kinetic_perturbations[:, k_param] = kinetic_perturbations_subset[:, j]

    kinetic_multipliers = np.exp(kinetic_perturbations)
    kinetic_multipliers_ct = kinetic_multipliers.dot(ct2rmg_matrix.T)


    # Cantera does well if you give it lots of CPUs for a single simulation
    # but slows down if you try to parallelize different simulations across multiple processes
    # so we run the simulations in serial here do the parallelize across SLURM array jobs.
    for i in range(CHUNK_SIZE):

        # perturb all the species
        for sp_index in g_params:
            # random perturbation
            perturbed_sp = ezuq.util.perturb_species_ct(gas.species()[sp_index], thermo_perturbations[i, sp_index])
            gas.modify_species(sp_index, perturbed_sp)

        # set multipliers
        for j, rxn_index_rmg in enumerate(k_params):
            ct_indices = np.where(ct2rmg_matrix[:, rxn_index_rmg] == 1)[0]
            for ct_index in ct_indices:
                gas.set_multiplier(kinetic_multipliers_ct[i, ct_index], ct_index)
        try:
            # TODO add timeout here so that if a simulation is taking too long we can skip it and move on 
            y[i, :] = run_simulation(gas, settings)
        except ct.CanteraError:
            y[i, :] = np.nan  # if the simulation fails, we can fill in NaNs and move on. The Morris analysis can handle some failed simulations as long as most of them work.

        # Reset things
        for sp_index in g_params:
            gas.modify_species(sp_index, thermo_copies[sp_index])
        gas.set_multiplier(1.0)

    np.save(output_filename, y)

def reassemble_chunks(condition_dir):
    """After all the chunks have been run, we need to reassemble the results into a single file for each condition"""

    condition_name = os.path.basename(condition_dir)
    y_files = sorted(glob.glob(os.path.join(condition_dir, 'monte_carlo_y', f'y_*.npy')))

    if len(y_files) == 0:
        raise ValueError('No files found')

    # test a sample
    sample_y = np.load(y_files[0])
    if sample_y.shape[0] != CHUNK_SIZE:
        raise ValueError(f"Expected chunk size of {CHUNK_SIZE} but got {sample_y.shape[0]} in file {y_files[0]}")

    k = sample_y.shape[1]

    # get the total number of samples from the file count
    N = CHUNK_SIZE * len(y_files)
    monte_carlo_y = np.zeros((N, k))

    for i in range(len(y_files)):
        match = re.search(r'y_(\d+).npy', y_files[i])
        index = int(match.group(1))
        data = np.load(y_files[i])
        assert data.shape == (CHUNK_SIZE, k)
        monte_carlo_y[index * CHUNK_SIZE: (index + 1) * CHUNK_SIZE, :] = data

    # see how many failed
    index_redo = set()
    invalid_count = 0
    for i in range(monte_carlo_y.shape[0]):
        if np.all(monte_carlo_y[i, :] == 0):
            invalid_count += 1
            index_redo.add(int(i / 1000.0))

    print(f'{condition_name}, {(monte_carlo_y.shape[0] - invalid_count) / monte_carlo_y.shape[0] * 100:.2f} % valid')
    if invalid_count / monte_carlo_y.shape[0] > 0.01:
        print(f'You should redo condition {condition_name}')
        print(f'Redo indices: {index_redo}')

    np.save(os.path.join(condition_dir, 'monte_carlo_results.npy'), monte_carlo_y)


if __name__ == "__main__":
    settings_yaml = sys.argv[1]
    chunk_index = int(sys.argv[2])
    run_chunk(settings_yaml, chunk_index)
