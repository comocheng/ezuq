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
import scipy.stats.qmc

import ezuq.util
from ezuq.simulation.jsr import run_simulation

CHUNK_SIZE = 1000

CONFIDENCE_INTERVAL = 0.9999999980268247  # 6 sigma


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

            if 'z_g_params' in morris_screen_result:
                # Not sure how to do a truncation in decomposed space
                raise NotImplementedError()

            else:  # we're truncating in physical parameter space
                g_params = morris_screen_result['g_params']
                k_params = morris_screen_result['k_params']
                g_param_names = morris_screen_result['g_param_names']
                k_param_names = morris_screen_result['k_param_names']

                alpha = float((1 - CONFIDENCE_INTERVAL) / 2)
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

                if not ezuq.util.is_diagonal(thermo_covariance_matrix) or not ezuq.util.is_diagonal(kinetic_covariance_matrix):
                    print('Correlated parameter matrix means Kucherenko 2012 sampling')
                    # can't use saltelli scheme, have to do this mostly from scratch
                    k = len(g_params) + len(k_params)
                    sampler = scipy.stats.qmc.Sobol(d=2*k, scramble=True, seed=SEED)
                    N_power = np.log2(N)
                    assert 2 ** N_power == int(N), 'N must be a power of 2'
                    u_all = sampler.random_base2(m=int(N_power))

                    # we're just going to generate u and u_prime here and save them as u_all. Then run_chunk will do all the calculations
                    print(f'Generated {u_all.shape[0]} samples with 2*{int(u_all.shape[1] / 2)} variables')

                    # this is saved in the condition dir because model reduction means you could have different parameters important at different conditions
                    np.save(os.path.join(sobol_condition_dir, 'sobol_samples_u_all.npy'), u_all)

                else:
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

        if not ezuq.util.is_diagonal(thermo_covariance_matrix) or not ezuq.util.is_diagonal(kinetic_covariance_matrix):
            # Kucherenko 2012 sampling setup

            # can't use saltelli scheme, have to do this mostly from scratch
            k = len(g_params) + len(k_params)
            sampler = scipy.stats.qmc.Sobol(d=2*k, scramble=True, seed=SEED)
            N_power = np.log2(N)
            assert 2 ** N_power == int(N), 'N must be a power of 2'
            u_all = sampler.random_base2(m=int(N_power))

            # we're just going to generate u and u_prime here and save them as u_all. Then run_chunk will do all the calculations
            print(f'Generated {u_all.shape[0]} samples with 2*{int(u_all.shape[1] / 2)} variables')
            np.save(os.path.join(sobol_dir, 'sobol_samples_u_all.npy'), u_all)
        else:
            # covariance matrices are independent, so sampling is straightforward transformation from uniform distribution to normal

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
        # figure out if we're doing model reduction
        if 'model_reduction' in problem and problem['model_reduction'] == True:
            print('Dependent parameters and model reduction')


            # Now figure out what style truncation we're doing. Correlated or uncorrelated

            if 'z_g_params' in problem:
                # truncate in decomposed space
                raise NotImplementedError('Truncating in decomposed space not implemented')

            else:
                print('Dependent parameters with model truncated in physical parameter space')
                results_dir = os.path.join(condition_dir, 'sobol_conditional_y')
                os.makedirs(results_dir, exist_ok=True)

                g_params = problem['g_params']
                k_params = problem['k_params']
                def f(x):
                    # this is the function that takes in the samples in normal space and outputs the simulation results.
                    # this should really be used in every one of these sampling functions
                    # TODO, refactor all code to use this. Put it in JSR
                    y = np.zeros((x.shape[0], gas.n_species))
                    thermo_perturbations_reduced = x[:, :len(g_params)] * 4184  # convert RMG-UQ's kcal/mol to J/mol
                    kinetic_perturbations_reduced = x[:, len(g_params):]  # these are the perturbations in log space, so we can exponentiate to get the kinetic multipliers
                    kinetic_multipliers_rmg_reduced = np.exp(kinetic_perturbations_reduced)

                    thermo_perturbations = np.zeros((x.shape[0], len(species_list)))
                    kinetic_multipliers_rmg = np.ones((x.shape[0], len(reaction_list)))
                    for i, sp_index in enumerate(g_params):
                        thermo_perturbations[:, sp_index] = thermo_perturbations_reduced[:, i]
                    for i, rxn_index in enumerate(k_params):
                        kinetic_multipliers_rmg[:, rxn_index] = kinetic_multipliers_rmg_reduced[:, i]
                    kinetic_multipliers_ct = kinetic_multipliers_rmg @ ct2rmg_matrix.T  # convert from RMG reaction space to Cantera reaction space
                    # TODO get rid of the RMG to cantera conversions. Just save a cantera covariance matrix as the input

                    for i in range(x.shape[0]):
                        # perturb all the species
                        for sp_index in g_params:
                            perturbed_sp = ezuq.util.perturb_species_ct(gas.species()[sp_index], thermo_perturbations[i, sp_index])
                            gas.modify_species(sp_index, perturbed_sp)

                        # set multipliers
                        for rxn_index_ct in range(gas.n_reactions):
                            if kinetic_multipliers_ct[i, rxn_index_ct] != 1.0:
                                gas.set_multiplier(kinetic_multipliers_ct[i, rxn_index_ct], rxn_index_ct)
                        try:
                            y[i, :] = run_simulation(gas, settings)
                        except ct.CanteraError:
                            y[i, :] = np.nan

                        # Reset things
                        for sp_index in g_params:
                            gas.modify_species(sp_index, thermo_copies[sp_index])
                        gas.set_multiplier(1.0)

                    output_species_index = settings['output_species_index']
                    return y[:, output_species_index]


                # Make the truncated thermo covariance matrix
                thermo_covariance_matrix_reduced = thermo_covariance_matrix[np.ix_(g_params, g_params)]
                kinetic_covariance_matrix_reduced = kinetic_covariance_matrix[np.ix_(k_params, k_params)]

                k = len(g_params) + len(k_params)

                # make a combined covariance matrix so we can do the sampling in one step
                cov = np.zeros((k, k))
                cov[:thermo_covariance_matrix_reduced.shape[0], :thermo_covariance_matrix_reduced.shape[0]] = thermo_covariance_matrix_reduced
                cov[thermo_covariance_matrix_reduced.shape[0]:, thermo_covariance_matrix_reduced.shape[0]:] = kinetic_covariance_matrix_reduced
                L = np.linalg.cholesky(cov)
                assert np.isclose(L @ L.T, cov).all()

                mean = np.zeros(k)

                u_all = np.load(os.path.join(condition_dir, 'sobol_samples_u_all.npy'))

                if chunk_index * CHUNK_SIZE >= u_all.shape[0]:
                    print(f"Chunk index {chunk_index} is out of range for number of samples {u_all.shape[0]} with chunk size {CHUNK_SIZE}")
                    exit(0)

                # reduce the uniform Sobol samples to their relevant chunk
                maximum_index = min((chunk_index + 1) * CHUNK_SIZE, u_all.shape[0])
                u_all = u_all[chunk_index * CHUNK_SIZE: maximum_index, :]
                N = u_all.shape[0]
                all_N_indices = np.arange(N)

                # 1. Partition the Sobol samples into u and u_prime
                u = u_all[:, :k]
                u_prime = u_all[:, k:]

                # 2. Generate unit normals from unit uniform Sobol samples
                x_tilde = scipy.stats.norm.ppf(u)
                x_tilde_prime = scipy.stats.norm.ppf(u_prime)

                x = x_tilde @ L.T + mean  # L does not equal its transpose, so you have to be really careful here. check that you're recreating the cov matrix
                x_prime = x_tilde_prime @ L.T + mean

                f_y_z = f(x)
                f_y_z_filename = os.path.join(results_dir, f'f_y_z_{chunk_index:04}.npy')
                np.save(f_y_z_filename, f_y_z)
                f_y_prime_z_prime = f(x_prime)
                f_y_prime_z_prime_filename = os.path.join(results_dir, f'f_y_prime_z_prime_{chunk_index:04}.npy')
                np.save(f_y_prime_z_prime_filename, f_y_prime_z_prime)
                # don't compute variance D until all chunks have been computed


                f_y_z_bar_prime = np.zeros((N, k))
                f_y_bar_prime_z = np.zeros((N, k))
                for q in range(k):
                    y_indices = [q]  # compute 1st order index on first parameter
                    z_indices = list(set(np.arange(k)) - set(y_indices))

                    # partition the mean and covariance matrices
                    # mean is zero for all of these
                    mu_y = mean[y_indices]
                    mu_z = mean[z_indices]
                    Sigma_y  = cov[np.ix_(y_indices, y_indices)]
                    Sigma_z  = cov[np.ix_(z_indices, z_indices)]
                    Sigma_yz = cov[np.ix_(y_indices, z_indices)]
                    Sigma_zy = cov[np.ix_(z_indices, y_indices)]

                    Sigma_y_inv = np.linalg.inv(Sigma_y)
                    Sigma_z_inv = np.linalg.inv(Sigma_z)

                    Sigma_zc = Sigma_z - Sigma_zy @ Sigma_y_inv @ Sigma_yz
                    Sigma_yc = Sigma_y - Sigma_yz @ Sigma_z_inv @ Sigma_zy

                    A_zc = np.linalg.cholesky(Sigma_zc)
                    A_yc = np.linalg.cholesky(Sigma_yc)

                    v_prime = u_prime[np.ix_(all_N_indices, y_indices)]
                    w_prime = u_prime[np.ix_(all_N_indices, z_indices)]

                    # 3. Generate unconditional normals
    
                    # split into subsets
                    y = x[np.ix_(all_N_indices, y_indices)]
                    z = x[np.ix_(all_N_indices, z_indices)]

                    # 4. Generate conditional normals
                    mu_zc = np.tile(mu_z.T, (N, 1)) + (Sigma_zy @ Sigma_y_inv @ (y - np.tile(mu_y.T, (N, 1))).T).T
                    mu_yc = np.tile(mu_y.T, (N, 1)) + (Sigma_yz @ Sigma_z_inv @ (z - np.tile(mu_z.T, (N, 1))).T).T

                    # partition the standard normals
                    y_tilde_prime = scipy.stats.norm.ppf(v_prime)
                    z_tilde_prime = scipy.stats.norm.ppf(w_prime)

                    y_bar_prime = y_tilde_prime @ A_yc.T + mu_yc
                    z_bar_prime = z_tilde_prime @ A_zc.T + mu_zc

                    # Have to reconstruct [y, z] using appropriate placement of indices
                    y_z_bar_prime = np.zeros((N, k))
                    y_z_bar_prime[:, y_indices] = y
                    y_z_bar_prime[:, z_indices] = z_bar_prime

                    y_bar_prime_z = np.zeros((N, k))
                    y_bar_prime_z[:, y_indices] = y_bar_prime
                    y_bar_prime_z[:, z_indices] = z

                    # Evaluate functions
                    f_y_z_bar_prime[:, q] = f(y_z_bar_prime)
                    f_y_bar_prime_z[:, q] = f(y_bar_prime_z)

                f_y_z_bar_prime_filename = os.path.join(results_dir, f'f_y_z_bar_prime_{chunk_index:04}.npy')
                np.save(f_y_z_bar_prime_filename, f_y_z_bar_prime)
                f_y_bar_prime_z_filename = os.path.join(results_dir, f'f_y_bar_prime_z_{chunk_index:04}.npy')
                np.save(f_y_bar_prime_z_filename, f_y_bar_prime_z)
















        else:
            # no model reduction
            # Kucherenko 2012 sampling method
            print('Dependent parameters with no model reduction')
            results_dir = os.path.join(condition_dir, 'sobol_conditional_y')
            os.makedirs(results_dir, exist_ok=True)

            def f(x):
                # this is the function that takes in the samples in normal space and outputs the simulation results.
                # this should really be used in every one of these sampling functions
                # TODO, refactor all code to use this. Put it in JSR
                y = np.zeros((x.shape[0], gas.n_species))
                thermo_perturbations = x[:, :thermo_covariance_matrix.shape[0]] * 4184  # convert RMG-UQ's kcal/mol to J/mol
                kinetic_perturbations = x[:, thermo_covariance_matrix.shape[0]:]  # these are the perturbations in log space, so we can exponentiate to get the kinetic multipliers
                kinetic_multipliers_rmg = np.exp(kinetic_perturbations)
                kinetic_multipliers_ct = kinetic_multipliers_rmg @ ct2rmg_matrix.T  # convert from RMG reaction space to Cantera reaction space

                for i in range(x.shape[0]):
                    # perturb all the species
                    for sp_index in range(gas.n_species):
                        perturbed_sp = ezuq.util.perturb_species_ct(gas.species()[sp_index], thermo_perturbations[i, sp_index])
                        gas.modify_species(sp_index, perturbed_sp)

                    # set multipliers
                    for rxn_index_ct in range(gas.n_reactions):
                        gas.set_multiplier(kinetic_multipliers_ct[i, rxn_index_ct], rxn_index_ct)
                    try:
                        y[i, :] = run_simulation(gas, settings)
                    except ct.CanteraError:
                        y[i, :] = np.nan

                    # Reset things
                    for sp_index in range(gas.n_species):
                        gas.modify_species(sp_index, thermo_copies[sp_index])
                    gas.set_multiplier(1.0)

                output_species_index = settings['output_species_index']
                return y[:, output_species_index]

            k = thermo_covariance_matrix.shape[0] + kinetic_covariance_matrix.shape[0]

            # make a combined covariance matrix so we can do the sampling in one step
            cov = np.zeros((k, k))
            cov[:thermo_covariance_matrix.shape[0], :thermo_covariance_matrix.shape[0]] = thermo_covariance_matrix
            cov[thermo_covariance_matrix.shape[0]:, thermo_covariance_matrix.shape[0]:] = kinetic_covariance_matrix
            L = np.linalg.cholesky(cov)
            assert np.isclose(L @ L.T, cov).all()

            mean = np.zeros(k)

            u_all = np.load(os.path.join(sobol_dir, 'sobol_samples_u_all.npy'))

            if chunk_index * CHUNK_SIZE >= u_all.shape[0]:
                print(f"Chunk index {chunk_index} is out of range for number of samples {u_all.shape[0]} with chunk size {CHUNK_SIZE}")
                exit(0)

            # reduce the uniform Sobol samples to their relevant chunk
            maximum_index = min((chunk_index + 1) * CHUNK_SIZE, u_all.shape[0])
            u_all = u_all[chunk_index * CHUNK_SIZE: maximum_index, :]
            N = u_all.shape[0]
            all_N_indices = np.arange(N)

            # 1. Partition the Sobol samples into u and u_prime
            u = u_all[:, :k]
            u_prime = u_all[:, k:]

            # 2. Generate unit normals from unit uniform Sobol samples
            x_tilde = scipy.stats.norm.ppf(u)
            x_tilde_prime = scipy.stats.norm.ppf(u_prime)

            x = x_tilde @ L.T + mean  # L does not equal its transpose, so you have to be really careful here. check that you're recreating the cov matrix
            x_prime = x_tilde_prime @ L.T + mean

            f_y_z = f(x)
            f_y_z_filename = os.path.join(results_dir, f'f_y_z_{chunk_index:04}.npy')
            np.save(f_y_z_filename, f_y_z)
            f_y_prime_z_prime = f(x_prime)
            f_y_prime_z_prime_filename = os.path.join(results_dir, f'f_y_prime_z_prime_{chunk_index:04}.npy')
            np.save(f_y_prime_z_prime_filename, f_y_prime_z_prime)
            # don't compute variance D until all chunks have been computed


            f_y_z_bar_prime = np.zeros((N, k))
            f_y_bar_prime_z = np.zeros((N, k))
            for q in range(k):
                y_indices = [q]  # compute 1st order index on first parameter
                z_indices = list(set(np.arange(k)) - set(y_indices))

                # partition the mean and covariance matrices
                # mean is zero for all of these
                mu_y = mean[y_indices]
                mu_z = mean[z_indices]
                Sigma_y  = cov[np.ix_(y_indices, y_indices)]
                Sigma_z  = cov[np.ix_(z_indices, z_indices)]
                Sigma_yz = cov[np.ix_(y_indices, z_indices)]
                Sigma_zy = cov[np.ix_(z_indices, y_indices)]

                Sigma_y_inv = np.linalg.inv(Sigma_y)
                Sigma_z_inv = np.linalg.inv(Sigma_z)

                Sigma_zc = Sigma_z - Sigma_zy @ Sigma_y_inv @ Sigma_yz
                Sigma_yc = Sigma_y - Sigma_yz @ Sigma_z_inv @ Sigma_zy

                A_zc = np.linalg.cholesky(Sigma_zc)
                A_yc = np.linalg.cholesky(Sigma_yc)

                v_prime = u_prime[np.ix_(all_N_indices, y_indices)]
                w_prime = u_prime[np.ix_(all_N_indices, z_indices)]

                # 3. Generate unconditional normals
 
                # split into subsets
                y = x[np.ix_(all_N_indices, y_indices)]
                z = x[np.ix_(all_N_indices, z_indices)]

                # 4. Generate conditional normals
                mu_zc = np.tile(mu_z.T, (N, 1)) + (Sigma_zy @ Sigma_y_inv @ (y - np.tile(mu_y.T, (N, 1))).T).T
                mu_yc = np.tile(mu_y.T, (N, 1)) + (Sigma_yz @ Sigma_z_inv @ (z - np.tile(mu_z.T, (N, 1))).T).T

                # partition the standard normals
                y_tilde_prime = scipy.stats.norm.ppf(v_prime)
                z_tilde_prime = scipy.stats.norm.ppf(w_prime)

                y_bar_prime = y_tilde_prime @ A_yc.T + mu_yc
                z_bar_prime = z_tilde_prime @ A_zc.T + mu_zc

                # Have to reconstruct [y, z] using appropriate placement of indices
                y_z_bar_prime = np.zeros((N, k))
                y_z_bar_prime[:, y_indices] = y
                y_z_bar_prime[:, z_indices] = z_bar_prime

                y_bar_prime_z = np.zeros((N, k))
                y_bar_prime_z[:, y_indices] = y_bar_prime
                y_bar_prime_z[:, z_indices] = z

                # Evaluate functions
                f_y_z_bar_prime[:, q] = f(y_z_bar_prime)
                f_y_bar_prime_z[:, q] = f(y_bar_prime_z)

            f_y_z_bar_prime_filename = os.path.join(results_dir, f'f_y_z_bar_prime_{chunk_index:04}.npy')
            np.save(f_y_z_bar_prime_filename, f_y_z_bar_prime)
            f_y_bar_prime_z_filename = os.path.join(results_dir, f'f_y_bar_prime_z_{chunk_index:04}.npy')
            np.save(f_y_bar_prime_z_filename, f_y_bar_prime_z)


def reassemble_chunks(condition_dir):
    """After all the chunks have been run, we need to reassemble the results into a single file for each condition"""

    condition_name = os.path.basename(condition_dir)
    sobol_dir = os.path.dirname(condition_dir)

    # Figure out what sort of analysis is being done here
    y_files = sorted(glob.glob(os.path.join(condition_dir, 'sobol_y', f'y_[0-9]*.npy')))
    f_y_z_files = sorted(glob.glob(os.path.join(condition_dir, 'sobol_conditional_y', f'f_y_z_[0-9]*.npy')))
    if f_y_z_files:
        print(f"Found files for conditional Sobol analysis for condition {condition_name}, reassembling those")
        # load the all_u file to get the number of samples and k
        if os.path.exists(os.path.join(condition_dir, 'sobol_samples_u_all.npy')):
            u_all = np.load(os.path.join(condition_dir, 'sobol_samples_u_all.npy'))
        else:
            u_all = np.load(os.path.join(sobol_dir, 'sobol_samples_u_all.npy'))  # no model reduction case, the samples are in the sobol_dir not the condition_dir
        N = u_all.shape[0]
        k = u_all.shape[1] // 2
        f_y_prime_z_prime_files = sorted(glob.glob(os.path.join(condition_dir, 'sobol_conditional_y', f'f_y_prime_z_prime_[0-9]*.npy')))
        f_y_z_bar_prime_files = sorted(glob.glob(os.path.join(condition_dir, 'sobol_conditional_y', f'f_y_z_bar_prime_[0-9]*.npy')))
        f_y_bar_prime_z_files = sorted(glob.glob(os.path.join(condition_dir, 'sobol_conditional_y', f'f_y_bar_prime_z_[0-9]*.npy')))

        f_y_z = np.zeros(N)
        f_y_prime_z_prime = np.zeros(N)
        f_y_z_bar_prime = np.zeros((N, k))
        f_y_bar_prime_z = np.zeros((N, k))
        for i in range(len(f_y_z_files)):
            match = re.search(r'f_y_z_(\d+).npy', f_y_z_files[i])
            if not match:
                raise ValueError(f"Could not extract chunk index from filename {f_y_z_files[i]}")
            index = int(match.group(1))
            data = np.load(f_y_z_files[i])
            assert data.shape == (min(CHUNK_SIZE, N - index * CHUNK_SIZE),), f"Expected shape of {(min(CHUNK_SIZE, N - index * CHUNK_SIZE),)} but got {data.shape} in file {f_y_z_files[i]}"
            f_y_z[index * CHUNK_SIZE: index * CHUNK_SIZE + data.shape[0]] = data

            data = np.load(f_y_prime_z_prime_files[i])
            assert data.shape == (min(CHUNK_SIZE, N - index * CHUNK_SIZE),), f"Expected shape of {(min(CHUNK_SIZE, N - index * CHUNK_SIZE),)} but got {data.shape} in file {f_y_prime_z_prime_files[i]}"
            f_y_prime_z_prime[index * CHUNK_SIZE: index * CHUNK_SIZE + data.shape[0]] = data

            data = np.load(f_y_z_bar_prime_files[i])
            assert data.shape == (min(CHUNK_SIZE, N - index * CHUNK_SIZE), k), f"Expected shape of {(min(CHUNK_SIZE, N - index * CHUNK_SIZE), k)} but got {data.shape} in file {f_y_z_bar_prime_files[i]}"
            f_y_z_bar_prime[index * CHUNK_SIZE: index * CHUNK_SIZE + data.shape[0], :] = data

            data = np.load(f_y_bar_prime_z_files[i])
            assert data.shape == (min(CHUNK_SIZE, N - index * CHUNK_SIZE), k), f"Expected shape of {(min(CHUNK_SIZE, N - index * CHUNK_SIZE), k)} but got {data.shape} in file {f_y_bar_prime_z_files[i]}"
            f_y_bar_prime_z[index * CHUNK_SIZE: index * CHUNK_SIZE + data.shape[0], :] = data
        np.save(os.path.join(condition_dir, 'f_y_z.npy'), f_y_z)
        np.save(os.path.join(condition_dir, 'f_y_prime_z_prime.npy'), f_y_prime_z_prime)
        np.save(os.path.join(condition_dir, 'f_y_z_bar_prime.npy'), f_y_z_bar_prime)
        np.save(os.path.join(condition_dir, 'f_y_bar_prime_z.npy'), f_y_bar_prime_z)

    if y_files:
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


def compute_sobol_indices(condition_dir):
    """After reassembling the chunks, we can compute the Sobol indices"""

    condition_name = os.path.basename(condition_dir)

    f_y_z_file = os.path.join(condition_dir, 'f_y_z.npy')
    if not os.path.exists(f_y_z_file):
        print(f"No conditional Sobol results found for condition {condition_name}, skipping index computation")
        return

    f_y_z = np.load(f_y_z_file)
    f_y_prime_z_prime = np.load(os.path.join(condition_dir, 'f_y_prime_z_prime.npy'))
    f_y_z_bar_prime = np.load(os.path.join(condition_dir, 'f_y_z_bar_prime.npy'))
    f_y_bar_prime_z = np.load(os.path.join(condition_dir, 'f_y_bar_prime_z.npy'))

    k = f_y_z_bar_prime.shape[1]

    D = np.var(f_y_z)

    S1 = np.zeros(k)
    ST = np.zeros(k)
    for q in range(k):
        S1_contributions = np.multiply(f_y_z, f_y_z_bar_prime[:, q] - f_y_prime_z_prime)
        ST_contributions = np.float_power(f_y_z - f_y_bar_prime_z[:, q], 2.0)

        S1[q] = np.mean(S1_contributions, axis=0) / D
        ST[q] = np.mean(ST_contributions, axis=0) / (2.0 * D)

    return S1, ST

if __name__ == "__main__":
    settings_yaml = sys.argv[1]
    chunk_index = int(sys.argv[2])
    run_chunk(settings_yaml, chunk_index)
