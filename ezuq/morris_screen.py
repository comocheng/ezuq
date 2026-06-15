"""Code for running initial Morris screening. For now we assume everyone is using the same Cord JSR experiment I am"""

# This code will need to read in the RMG uncertainty matrices.

import os
import re
import glob
import pickle
import shutil
import sys

import cantera as ct
import numpy as np
import rmgpy.chemkin
import SALib.sample.morris
import scipy.stats
import yaml

import ezuq.util
from ezuq.simulation.jsr import run_simulation

CHUNK_SIZE = 1000

def setup_runfiles(working_dir, conditions, N_SAMPLES=100, NUM_LEVELS=4, SEED=400):
    """Set up the runfiles for the Morris screening.
    working_dir should be the directory where the RMG and Cantera mechanisms are saved.
    """

    morris_dir = os.path.join(working_dir, 'morris_screen')
    os.makedirs(morris_dir, exist_ok=True)

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
    with open(os.path.join(morris_dir, 'problem_desc.yaml'), 'w') as f:
        yaml.dump(problem, f)

    # Generate Morris samples (takes a minute)
    X = SALib.sample.morris.sample(problem, N=N_SAMPLES, num_levels=NUM_LEVELS, seed=SEED)
    print(f'Generated {X.shape[0]} samples with {X.shape[1]} variables')
    np.save(os.path.join(morris_dir, 'morris_samples.npy'), X)

    ezuq.util.setup_condition_dirs(morris_dir, conditions)

    # copy the slurm script into the Morris dir
    shutil.copyfile(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'SLURM', 'run_morris.sh'), os.path.join(morris_dir, 'run_morris.sh'))


def run_chunk(settings_yaml, chunk_index):
    """Run a chunk of the morris simulations
    Assumes the following directory structure:
    working_dir/
        chem_annotated.inp
        species_dictionary.txt
        chem_annotated.yaml
        ct2rmg_rxn.pickle
        thermo_covariance_matrix.npy
        kinetic_covariance_matrix.npy
        morris_screen/
            problem_desc.yaml
            morris_samples.npy
            550K/
                settings.yaml
                morris_y/
            650K/
                settings.yaml
                morris_y/
            750K/
                settings.yaml
                morris_y/
    """
    
    condition_dir = os.path.dirname(os.path.abspath(settings_yaml))
    morris_dir = os.path.dirname(condition_dir)
    working_dir = os.path.dirname(morris_dir)
    results_dir = os.path.join(condition_dir, 'morris_y')
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

    with open(os.path.join(morris_dir, 'problem_desc.yaml'), 'r') as f:
        problem = yaml.load(f, Loader=yaml.FullLoader)
    assert problem['num_vars'] == len(species_list) + len(reaction_list), "Problem description number of variables does not match number of species + reactions"

    perturbations = np.load(os.path.join(morris_dir, 'morris_samples.npy'))
    if chunk_index * CHUNK_SIZE >= perturbations.shape[0]:
        raise ValueError(f"Chunk index {chunk_index} is out of range for number of samples {perturbations.shape[0]} with chunk size {CHUNK_SIZE}")
    assert perturbations.shape[0] / CHUNK_SIZE < 1000  # this exceeds the array size max we can use for SLURM 1000 array for a chunk of 1000
    assert perturbations.shape[1] == problem['num_vars']

    # -------------------- Load Morris perturbations and convert from unit uniform to actual perturbations using Nataf Transformation --------------------
    maximum_index = min((chunk_index + 1) * CHUNK_SIZE, perturbations.shape[0])
    perturbations_chunk = perturbations[chunk_index * CHUNK_SIZE: maximum_index, :]

    # ------------- Thermo perturbations -------------
    thermo_uniform_perturbations = perturbations_chunk[:, :len(species_list)]
    L_thermo = np.linalg.cholesky(thermo_covariance_matrix)
    assert np.isclose(L_thermo @ L_thermo.T, thermo_covariance_matrix).all()
    z_thermo = scipy.stats.norm.ppf(thermo_uniform_perturbations)
    thermo_perturbations = (L_thermo @ z_thermo.T).T * 4184  # convert RMG-UQ's kcal/mol to J/mol

    # ------------- Kinetic perturbations -------------
    kinetic_uniform_perturbations = perturbations_chunk[:, len(species_list):]
    L_kinetic = np.linalg.cholesky(kinetic_covariance_matrix)
    assert np.isclose(L_kinetic @ L_kinetic.T, kinetic_covariance_matrix).all()
    z_kinetic = scipy.stats.norm.ppf(kinetic_uniform_perturbations)
    kinetic_perturbations = (L_kinetic @ z_kinetic.T).T  # these are the perturbations in log space, so we can exponentiate to get the kinetic multipliers
    kinetic_multipliers_rmg = np.exp(kinetic_perturbations)
    kinetic_multipliers_ct = kinetic_multipliers_rmg @ ct2rmg_matrix.T  # convert from RMG reaction space to Cantera reaction space

    # save copies of all thermo for faster perturbation
    thermo_copies = []
    for sp_index in range(gas.n_species):
        thermo_copies.append(ct.Species().from_dict(gas.species()[sp_index].input_data.copy()))

    # Run simulations
    y = np.zeros((CHUNK_SIZE, gas.n_species))

    # Cantera does well if you give it lots of CPUs for a single simulation
    # but slows down if you try to parallelize different simulations across multiple processes
    # so we run the simulations in serial here do the parallelize across SLURM array jobs.
    for i in range(perturbations_chunk.shape[0]):

        # perturb all the species
        for sp_index in range(gas.n_species):
            # random perturbation
            if thermo_perturbations[i, sp_index] != 0:
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


def save_reduced_set(working_dir, conditions, i_sens, mu_star_threshold=0.05, verbose=True):
    """After we analyze the Morris results, we can save the reduced set of parameters that we want to use for the full global sampling
    
    mu_star_threshold is the threshold for considering a parameter to be irrelevant. If a parmeter's mu_star is less than mu_star_threshold * max(mu_star), it gets left out of the reduced set
    """
    
    morris_dir = os.path.join(working_dir, 'morris_screen')
    morris_samples = np.load(os.path.join(morris_dir, 'morris_samples.npy'))
    with open(os.path.join(morris_dir, 'problem_desc.yaml'), 'rb') as f:
        problem = yaml.load(f, Loader=yaml.FullLoader)

    chemkin_file = os.path.join(working_dir, 'chem_annotated.inp')
    dictionary_file = os.path.join(working_dir, 'species_dictionary.txt')
    species_list, reaction_list = rmgpy.chemkin.load_chemkin_file(chemkin_file, dictionary_file)

    gas = ct.Solution(os.path.join(working_dir, 'chem_annotated.yaml'))
    
    # Do analysis in physical parameter space
    # load the covariance matrices
    thermo_covariance_matrix = np.load(os.path.join(working_dir, 'thermo_covariance_matrix.npy'))
    kinetic_covariance_matrix = np.load(os.path.join(working_dir, 'kinetic_covariance_matrix.npy'))

    # Transform the Morris samples from unit uniform to standard normal
    z = scipy.stats.norm.ppf(morris_samples)
    z_thermo = z[:, :gas.n_species]
    z_kinetic = z[:, gas.n_species:]
    
    L_thermo = np.linalg.cholesky(thermo_covariance_matrix)
    assert np.isclose(L_thermo @ L_thermo.T, thermo_covariance_matrix).all()
    
    L_kinetic = np.linalg.cholesky(kinetic_covariance_matrix)
    assert np.isclose(L_kinetic @ L_kinetic.T, kinetic_covariance_matrix).all()

    # Transform the Morris samples from standard normal to the actual perturbations using the Cholesky decomposition of the covariance matrix
    w_thermo = (L_thermo @ z_thermo.T).T
    w_kinetic = (L_kinetic @ z_kinetic.T).T
    w = np.concatenate((w_thermo, w_kinetic), axis=1)

    if isinstance(conditions, dict):
        conditions = [conditions]

    contribution_results = {}

    for condition in conditions:
        morris_y = np.load(os.path.join(morris_dir, condition['name'], 'morris_sim_results.npy'))
        nominal_values = ezuq.simulation.jsr.run_simulation(gas, condition)
        for i in range(morris_y.shape[0]):
            if np.isnan(morris_y[i, :]).any():
                morris_y[i, :] = nominal_values

        if not ezuq.util.is_diagonal(thermo_covariance_matrix) or not ezuq.util.is_diagonal(kinetic_covariance_matrix):
            raise NotImplementedError()
        else:
            # parameters are independent. reduce in physical parameter space (as opposed to decomposed space)
            physical_result = SALib.analyze.morris.analyze(problem, w, morris_y[:, i_sens], scaled=True)
            # independent_result = SALib.analyze.morris.analyze(problem, z, morris_y[:, i_sens], scaled=False)

            # Rank parameter names by mean effect
            contributions = [(x, y) for y, x in sorted(zip(physical_result['mu_star'].data, problem['names']))][::-1]
            
            # define the tolerance for considering a parameter to be irrelevant
            threshold = mu_star_threshold * contributions[0][1]  # use 0.01 for tighter tolerance
            k_params = []
            g_params = []
            
            species_names = [x.to_chemkin() for x in species_list]
            reaction_names = [x.to_chemkin(species_list, kinetics=False) for x in reaction_list]
            for i in range(len(contributions)):
                if contributions[i][1] < threshold:
                    if verbose:
                        print(f'Reduced to {i} params')
                    break
            
                name = contributions[i][0]
                if verbose:
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

            contribution_results[condition['name']] = physical_result
    return contribution_results


def get_results_for_morris_screen(gas, settings):
    """Do the Morris Analysis and return the contributions in sorted order"""
    raise NotImplementedError()  # this is not ready yet, we need to reassemble the results from the chunks first and then we can do the analysis. For now just return the problem_desc for the reduced set?
    # # Just return the problem_desc for the reduced set?

    # # load the covariance matrices
    # thermo_covariance_matrix = np.load(os.path.join(working_dir, 'thermo_covariance_matrix.npy'))
    # kinetic_covariance_matrix = np.load(os.path.join(working_dir, 'kinetic_covariance_matrix.npy'))

    # z = scipy.stats.norm.ppf(morris_samples)
    # z_thermo = z[:, :gas.n_species]
    # z_kinetic = z[:, gas.n_species:]


    # L_thermo = np.linalg.cholesky(thermo_covariance_matrix)
    # assert np.isclose(L_thermo @ L_thermo.T, thermo_covariance_matrix).all()
    # thermo_perturbations = (L_thermo @ z_thermo.T).T * 4184  # convert RMG-UQ's kcal/mol to J/mol


    # L_kinetic = np.linalg.cholesky(kinetic_covariance_matrix)
    # assert np.isclose(L_kinetic @ L_kinetic.T, kinetic_covariance_matrix).all()
    # kinetic_perturbations = (L_kinetic @ z_kinetic.T).T  # in lnk space

    # w_thermo = (L_thermo @ z_thermo.T).T
    # w_kinetic = (L_kinetic @ z_kinetic.T).T
    # w = np.concatenate((w_thermo, w_kinetic), axis=1)


    # physical_result = SALib.analyze.morris.analyze(problem, w, morris_y[:, i_sens], scaled=True)



def _setup_runfiles_intermediate(working_dir, conditions, N_SAMPLES=100, NUM_LEVELS=4, SEED=400):
    """Like setup_runfiles, but for the intermediate thermo/kinetic parameters in the model

    working_dir should be the directory where the RMG and Cantera mechanisms are saved.
    Also requires sigma_qq_kinetics.npy, sigma_qq_thermo.npy, dG_dq.npy, and dlnk_dq to be in working_dir
    """
    raise NotImplementedError()

    # morris_q_dir = os.path.join(working_dir, 'morris_screen_q')
    # os.makedirs(morris_q_dir, exist_ok=True)

    # # load the covariance matrices
    # sigma_qq_thermo = np.load(os.path.join(working_dir, 'sigma_qq_thermo.npy'))
    # sigma_qq_kinetics = np.load(os.path.join(working_dir, 'sigma_qq_kinetics.npy'))

    # dG_dq = np.load(os.path.join(working_dir, 'dG_dq.npy'))
    # dlnk_dq = np.load(os.path.join(working_dir, 'dlnk_dq.npy'))

    # # confirm this matches the RMG mechanism
    # chemkin_file = os.path.join(working_dir, 'chem_annotated.inp')
    # dictionary_file = os.path.join(working_dir, 'species_dictionary.txt')
    # species_list, reaction_list = rmgpy.chemkin.load_chemkin_file(chemkin_file, dictionary_file)
    # assert len(species_list) == thermo_covariance_matrix.shape[0], "Thermo covariance matrix size does not match number of species"
    # assert len(reaction_list) == kinetic_covariance_matrix.shape[0], "Kinetic covariance matrix size does not match number of reactions"

    # cantera_file = os.path.join(working_dir, 'chem_annotated.yaml')
    # gas = ct.Solution(cantera_file)
    # with open(os.path.join(working_dir, 'ct2rmg_rxn.pickle'), 'rb') as f:
    #     ct2rmg_rxn = pickle.load(f)
    
    # assert gas.n_species == thermo_covariance_matrix.shape[0], "Thermo covariance matrix size does not match number of species in Cantera mechanism"
    # assert gas.n_reactions == len(ct2rmg_rxn), "Kinetic covariance matrix size does not match number of reactions in Cantera mechanism"
    # assert len(set(ct2rmg_rxn.values())) == len(reaction_list), "Reactions in Cantera mechanism do not match reactions in RMG mechanism"

    # # Define the problem using SALib format
    # # we need to clip the bounds to avoid infinity in the transformation to normal space.
    # confidence_interval = 0.95
    # alpha = (1 - confidence_interval) / 2

    # problem = {
    #     'num_vars': len(species_list) + len(reaction_list),
    #     'names': [sp.to_chemkin() for sp in species_list] + 
    #              [rxn.to_chemkin(species_list, kinetics=False) for rxn in reaction_list],
    #     'bounds': [[alpha, 1 - alpha]] * (len(species_list) + len(reaction_list)),  # (slightly clipped) unit uniforms, we'll handle the actual translation to valid perturbations later on
    # }
    # with open(os.path.join(morris_dir, 'problem_desc.yaml'), 'w') as f:
    #     yaml.dump(problem, f)

    # # Generate Morris samples (takes a minute)
    # X = SALib.sample.morris.sample(problem, N=N_SAMPLES, num_levels=NUM_LEVELS, seed=SEED)
    # print(f'Generated {X.shape[0]} samples with {X.shape[1]} variables')
    # np.save(os.path.join(morris_dir, 'morris_samples.npy'), X)

    # ezuq.util.setup_condition_dirs(morris_dir, conditions)

    # # copy the slurm script into the Morris dir
    # shutil.copyfile(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'SLURM', 'run_morris.sh'), os.path.join(morris_dir, 'run_morris.sh'))


def reassemble_chunks(morris_dir):
    """After all the chunks have been run, we need to reassemble the results into a single file for each condition"""

    # find all settings.yaml files in the condition dirs

    condition_dirs = [os.path.dirname(os.path.abspath(x)) for x in glob.glob(os.path.join(morris_dir, '*', 'settings.yaml'))]
    for condition_dir in condition_dirs:
        condition_name = os.path.basename(condition_dir)
        y_files = sorted(glob.glob(os.path.join(condition_dir, 'morris_y', f'y_*.npy')))

        if len(y_files) == 0:
            print(f'Skipping {condition_name} because no files')
            continue
        
        # test a sample
        sample_y = np.load(y_files[0])
        if sample_y.shape[0] != CHUNK_SIZE:
            raise ValueError(f"Expected chunk size of {CHUNK_SIZE} but got {sample_y.shape[0]} in file {y_files[0]}")
        
        k = sample_y.shape[1]

        # get the total number of samples from the morris samples file
        morris_samples = np.load(os.path.join(morris_dir, 'morris_samples.npy'))
        N = morris_samples.shape[0]
        morris_y = np.zeros((N, k))

        for i in range(len(y_files)):
            match = re.search(r'y_(\d+).npy', y_files[i])
            index = int(match.group(1))
            data = np.load(y_files[i])
            assert data.shape == (CHUNK_SIZE, k)
            try:
                morris_y[index * CHUNK_SIZE: (index + 1) * CHUNK_SIZE, :] = data
            except ValueError:
                fillshape = morris_y[index * CHUNK_SIZE:, :].shape
                morris_y[index * CHUNK_SIZE:, :] = data[:fillshape[0], :fillshape[1]]

        # see how many failed
        index_redo = set()
        invalid_count = 0
        for i in range(morris_y.shape[0]):
            if np.all(morris_y[i, :] == 0):
                invalid_count += 1
                index_redo.add(int(i / 1000.0))

        print(f'{condition_name}, {(morris_y.shape[0] - invalid_count) / morris_y.shape[0] * 100:.2f} % valid')
        if invalid_count / morris_y.shape[0] > 0.01:
            print(f'You should redo condition {condition_name}')
            print(f'Redo indices: {index_redo}')

        np.save(os.path.join(condition_dir, 'morris_sim_results.npy'), morris_y)


if __name__ == "__main__":
    settings_yaml = sys.argv[1]
    chunk_index = int(sys.argv[2])
    run_chunk(settings_yaml, chunk_index)
