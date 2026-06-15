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

CONFIDENCE_INTERVAL = 0.95

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
        # We are performing model reduction
        print('Using Morris screening results to reduce the parameter space for Sobol sampling')

        for condition in conditions:
            morris_condition_dir = os.path.join(morris_dir, condition['name'])
            sobol_condition_dir = os.path.join(sobol_dir, condition['name'])
            shutil.copyfile(os.path.join(morris_condition_dir, 'morris_screen_set.yaml'), os.path.join(sobol_condition_dir, 'morris_screen_set.yaml'))
            with open(os.path.join(sobol_condition_dir, 'morris_screen_set.yaml'), 'r') as f:
                morris_screen_result = yaml.load(f, Loader=yaml.FullLoader)

            g_params = morris_screen_result['g_params']
            k_params = morris_screen_result['k_params']
            g_param_names = morris_screen_result['g_param_names']
            k_param_names = morris_screen_result['k_param_names']

            alpha = (1 - CONFIDENCE_INTERVAL) / 2
            problem = {
                'num_vars': len(g_params) + len(k_params),
                'names': g_param_names + k_param_names,
                'bounds': [[alpha, 1 - alpha]] * (len(g_params) + len(k_params)),  # (slightly clipped) unit uniforms, we'll handle the actual translation to valid perturbations later on
                'g_params': g_params,
                'k_params': k_params,
                'g_param_names': g_param_names,
                'k_param_names': k_param_names,
                'model_reduction': True
            }
            with open(os.path.join(sobol_condition_dir, 'problem_desc.yaml'), 'w') as f:
                yaml.dump(problem, f, default_flow_style=False)

            # Generate Sobol samples (takes a minute)
            # these will be different for each condition because the reduced parameter set will be different
            X = SALib.sample.sobol.sample(problem, N=N, calc_second_order=False, seed=SEED)
            print(f'Generated {X.shape[0]} samples with {X.shape[1]} variables')
            np.save(os.path.join(sobol_condition_dir, 'sobol_samples.npy'), X)

    else:
        print('No reduced model params provided, this will run sampling on the full set')

        g_params = list(range(len(species_list)))
        k_params = list(range(len(reaction_list)))

        species_names = [x.to_chemkin() for x in species_list]
        reaction_names = [x.to_chemkin(species_list, kinetics=False) for x in reaction_list]

        # Define the problem using SALib format
        # we need to clip the bounds to avoid infinity in the transformation to normal space.
        alpha = (1 - CONFIDENCE_INTERVAL) / 2

        problem = {
            'num_vars': len(species_list) + len(reaction_list),
            'names': species_names + reaction_names,
            'bounds': [[alpha, 1 - alpha]] * (len(species_list) + len(reaction_list)),  # (slightly clipped) unit uniforms, we'll handle the actual translation to valid perturbations later on
            'g_params': g_params,
            'k_params': k_params,
            'g_param_names': species_names,
            'k_param_names': reaction_names,
            'model_reduction': False
        }
        for condition in conditions:
            sobol_condition_dir = os.path.join(sobol_dir, condition['name'])
            with open(os.path.join(sobol_condition_dir, 'problem_desc.yaml'), 'w') as f:
                yaml.dump(problem, f, default_flow_style=False)

        # Generate Sobol samples (takes a minute)
        X = SALib.sample.sobol.sample(problem, N=N, calc_second_order=False, seed=SEED)
        print(f'Generated {X.shape[0]} samples with {X.shape[1]} variables')
        np.save(os.path.join(sobol_dir, 'sobol_samples_unreduced.npy'), X)

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

    # save copies of all thermo for faster perturbation
    thermo_copies = []
    for sp_index in range(gas.n_species):
        thermo_copies.append(ct.Species().from_dict(gas.species()[sp_index].input_data.copy()))

    problem_desc_file = os.path.join(condition_dir, 'problem_desc.yaml')
    with open(problem_desc_file, 'r') as f:
        problem = yaml.load(f, Loader=yaml.FullLoader)

    if ezuq.util.is_diagonal(thermo_covariance_matrix) and ezuq.util.is_diagonal(kinetic_covariance_matrix):
        print('Parameters are independent')
        # Figure out if there was any model reduction done
        if 'model_reduction' in problem and problem['model_reduction'] == True:
            sobol_condition_dir = os.path.dirname(problem_desc_file)

            X = np.load(os.path.join(sobol_condition_dir, 'sobol_samples.npy'))
            if chunk_index * CHUNK_SIZE >= X.shape[0]:
                print(f"Chunk index {chunk_index} is out of range for number of samples {X.shape[0]} with chunk size {CHUNK_SIZE}")
                exit(0)
            
            # Get our subset of samples for this chunk
            maximum_index = min((chunk_index + 1) * CHUNK_SIZE, X.shape[0])
            X = X[chunk_index * CHUNK_SIZE: maximum_index, :]

            g_params = problem['g_params']
            k_params = problem['k_params']
            assert X.shape[1] == len(g_params) + len(k_params), "Number of variables in Sobol samples does not match number of species + reactions"


            # Get the thermo perturbations
            # ------------- Thermo perturbations -------------
            thermo_uniform_perturbations = X[:, :len(g_params)]
            L_thermo = np.linalg.cholesky(thermo_covariance_matrix)
            assert np.isclose(L_thermo @ L_thermo.T, thermo_covariance_matrix).all()
            z_thermo_reduced = scipy.stats.norm.ppf(thermo_uniform_perturbations)  # transform the unit uniforms to standard normals
            z_thermo = np.zeros((z_thermo_reduced.shape[0], len(species_list)))

            for i, sp_index in enumerate(g_params):
                z_thermo[:, sp_index] = z_thermo_reduced[:, i]

            thermo_perturbations = (L_thermo @ z_thermo.T).T * 4184  # convert RMG-UQ's kcal/mol to J/mol

            # ------------- Kinetic perturbations -------------
            kinetic_uniform_perturbations = X[:, len(g_params):]
            L_kinetic = np.linalg.cholesky(kinetic_covariance_matrix)
            assert np.isclose(L_kinetic @ L_kinetic.T, kinetic_covariance_matrix).all()
            z_kinetic_reduced = scipy.stats.norm.ppf(kinetic_uniform_perturbations)
            z_kinetic = np.zeros((z_kinetic_reduced.shape[0], len(reaction_list)))
            for i, rxn_index in enumerate(k_params):
                z_kinetic[:, rxn_index] = z_kinetic_reduced[:, i]

            kinetic_perturbations = (L_kinetic @ z_kinetic.T).T  # these are the perturbations in log space, so we can exponentiate to get the kinetic multipliers
            kinetic_multipliers_rmg = np.exp(kinetic_perturbations)
            kinetic_multipliers_ct = kinetic_multipliers_rmg @ ct2rmg_matrix.T  # convert from RMG reaction space to Cantera reaction space

            k_params_ct = np.where(kinetic_multipliers_ct[0, :] != 0)[0]
            k_params_ct_check = np.where(kinetic_multipliers_ct[1, :] != 0)[0]
            assert set(k_params_ct) == set(k_params_ct_check)
            # Do the perturbations and run the simulations
            # save copies of all thermo for faster perturbation
            thermo_copies = []
            for sp_index in range(gas.n_species):
                thermo_copies.append(ct.Species().from_dict(gas.species()[sp_index].input_data.copy()))

            # Cantera does well if you give it lots of CPUs for a single simulation
            # but slows down if you try to parallelize different simulations across multiple processes
            # so we run the simulations in serial here do the parallelize across SLURM array jobs.
            for i in range(X.shape[0]):

                # perturb all the species
                for sp_index in g_params:
                    perturbed_sp = ezuq.util.perturb_species_ct(gas.species()[sp_index], thermo_perturbations[i, sp_index])
                    gas.modify_species(sp_index, perturbed_sp)

                # set multipliers
                for rxn_index_ct in k_params_ct:
                    gas.set_multiplier(kinetic_multipliers_ct[i, rxn_index_ct], rxn_index_ct)
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
        else:
            # ------------- no model reduction, full sampling -------------
            X = np.load(os.path.join(sobol_dir, 'sobol_samples_unreduced.npy'))
            assert X.shape[1] == thermo_covariance_matrix.shape[0] + kinetic_covariance_matrix.shape[0], "Number of variables in Sobol samples does not match number of species + reactions"

            if chunk_index * CHUNK_SIZE >= X.shape[0]:
                print(f"Chunk index {chunk_index} is out of range for number of samples {X.shape[0]} with chunk size {CHUNK_SIZE}")
                exit(0)
    
            # Get our subset of samples for this chunk
            maximum_index = min((chunk_index + 1) * CHUNK_SIZE, X.shape[0])
            X = X[chunk_index * CHUNK_SIZE: maximum_index, :]

            # Get the thermo perturbations
            # ------------- Thermo perturbations -------------
            thermo_uniform_perturbations = X[:, :len(species_list)]
            L_thermo = np.linalg.cholesky(thermo_covariance_matrix)
            assert np.isclose(L_thermo @ L_thermo.T, thermo_covariance_matrix).all()
            z_thermo = scipy.stats.norm.ppf(thermo_uniform_perturbations)  # transform the unit uniforms to standard normals
            thermo_perturbations = (L_thermo @ z_thermo.T).T * 4184  # convert RMG-UQ's kcal/mol to J/mol

            # ------------- Kinetic perturbations -------------
            kinetic_uniform_perturbations = X[:, len(species_list):]
            L_kinetic = np.linalg.cholesky(kinetic_covariance_matrix)
            assert np.isclose(L_kinetic @ L_kinetic.T, kinetic_covariance_matrix).all()
            z_kinetic = scipy.stats.norm.ppf(kinetic_uniform_perturbations)
            kinetic_perturbations = (L_kinetic @ z_kinetic.T).T  # these are the perturbations in log space, so we can exponentiate to get the kinetic multipliers
            kinetic_multipliers_rmg = np.exp(kinetic_perturbations)
            kinetic_multipliers_ct = kinetic_multipliers_rmg @ ct2rmg_matrix.T  # convert from RMG reaction space to Cantera reaction space

            # Do the perturbations and run the simulations
            # save copies of all thermo for faster perturbation
            thermo_copies = []
            for sp_index in range(gas.n_species):
                thermo_copies.append(ct.Species().from_dict(gas.species()[sp_index].input_data.copy()))

            # Cantera does well if you give it lots of CPUs for a single simulation
            # but slows down if you try to parallelize different simulations across multiple processes
            # so we run the simulations in serial here do the parallelize across SLURM array jobs.
            for i in range(X.shape[0]):

                # perturb all the species
                for sp_index in range(gas.n_species):
                    # random perturbation
                    if thermo_perturbations[i, sp_index] != 0:  # not sure I need this. We're sampling everything...
                        perturbed_sp = ezuq.util.perturb_species_ct(gas.species()[sp_index], thermo_perturbations[i, sp_index])
                        gas.modify_species(sp_index, perturbed_sp)

                # set multipliers
                for rxn_index_ct in range(gas.n_reactions):
                    gas.set_multiplier(kinetic_multipliers_ct[i, rxn_index_ct], rxn_index_ct)
                try:
                    # TODO add timeout here so that if a simulation is taking too long we can skip it and move on 
                    y[i, :] = run_simulation(gas, settings)
                except ct.CanteraError:
                    y[i, :] = np.nan  # if the simulation fails, we can fill in NaNs and move on. The Morris analysis can handle some failed simulations as long as most of them work.

                # Reset things
                for sp_index in range(gas.n_species):
                    if thermo_perturbations[i, sp_index] != 0:
                        gas.modify_species(sp_index, thermo_copies[sp_index])
                gas.set_multiplier(1.0)

            np.save(output_filename, y)


    else:
        # Covariance matrices are not diagonal/independent
        raise NotImplementedError()


def reassemble_chunks(condition_dir):
    """After all the chunks have been run, we need to reassemble the results into a single file for each condition"""

    condition_name = os.path.basename(condition_dir)
    y_files = sorted(glob.glob(os.path.join(condition_dir, 'sobol_y', f'y_*.npy')))

    if len(y_files) == 0:
        print(f"No chunk files found for condition {condition_name} in {os.path.join(condition_dir, 'sobol_y')}")
        return

    # test a sample
    sample_y = np.load(y_files[0])
    if sample_y.shape[0] != CHUNK_SIZE:
        raise ValueError(f"Expected chunk size of {CHUNK_SIZE} but got {sample_y.shape[0]} in file {y_files[0]}")

    k = sample_y.shape[1]

    # get the total number of samples from the file count
    N = CHUNK_SIZE * len(y_files)
    sobol_y = np.zeros((N, k))

    for i in range(len(y_files)):
        match = re.search(r'y_(\d+).npy', y_files[i])
        index = int(match.group(1))
        data = np.load(y_files[i])
        assert data.shape == (CHUNK_SIZE, k)
        sobol_y[index * CHUNK_SIZE: (index + 1) * CHUNK_SIZE, :] = data

    # starting from the end, delete any rows that are still all zeros, the difference between chunks that were fully filled and the last chunk
    while np.all(sobol_y[-1, :] == 0):
        sobol_y = sobol_y[:-1, :]

    print(f'After reassembling, got {sobol_y.shape[0]} samples for condition {condition_name}')
    np.save(os.path.join(condition_dir, 'sobol_results.npy'), sobol_y)


if __name__ == "__main__":
    settings_yaml = sys.argv[1]
    chunk_index = int(sys.argv[2])
    run_chunk(settings_yaml, chunk_index)
