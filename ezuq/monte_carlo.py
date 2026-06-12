"""A module for running Monte Carlo samples of reaction model"""


import os
import sys
import pickle
import shutil
import numpy as np
import yaml
import cantera as ct
import rmgpy.chemkin

import ezuq.util
from ezuq.simulation.jsr import run_simulation

CHUNK_SIZE = 1000

def setup_runfiles(working_dir, conditions, morris_dir=None):
    """Set up the runfiles for Monte Carlo Sampling
    working_dir should be the directory where the RMG and Cantera mechanisms are saved.

    optional morris_dir to use the results of screening
    if no morris_dir is provided, this will set up the runfiles for a full Monte Carlo sampling, which is probably too expensive to run to convergence
    """

    monte_carlo_dir = os.path.join(working_dir, 'monte_carlo')
    os.makedirs(monte_carlo_dir, exist_ok=True)

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

    if morris_dir is None:
        # Define the problem using SALib format
        problem = {
            'num_vars': len(species_list) + len(reaction_list),
            'names': [sp.to_chemkin() for sp in species_list] + 
                    [rxn.to_chemkin(species_list, kinetics=False) for rxn in reaction_list],
        }
        with open(os.path.join(monte_carlo_dir, 'problem_desc.yaml'), 'w') as f:
            yaml.dump(problem, f)

        ezuq.util.setup_condition_dirs(monte_carlo_dir, conditions)

        # copy the slurm script into the Monte Carlo dir
        shutil.copyfile(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'SLURM', 'run_monte_carlo.sh'), os.path.join(monte_carlo_dir, 'run_monte_carlo.sh'))
    else:
        raise NotImplementedError("Setting up runfiles from a previous Morris screening is not implemented yet")

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
        monte_carlo/
            problem_desc.yaml
            550K/
                settings.yaml
                monte_carlo_y/
            650K/
                settings.yaml
                monte_carlo_y/
            750K/
                settings.yaml
                monte_carlo_y/
    """
    
    condition_dir = os.path.dirname(os.path.abspath(settings_yaml))
    monte_carlo_dir = os.path.dirname(condition_dir)
    working_dir = os.path.dirname(monte_carlo_dir)
    results_dir = os.path.join(condition_dir, 'monte_carlo_y')
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

    with open(os.path.join(monte_carlo_dir, 'problem_desc.yaml'), 'r') as f:
        problem = yaml.load(f, Loader=yaml.FullLoader)


    y = np.zeros((CHUNK_SIZE, gas.n_species))
    if problem['num_vars'] == len(species_list) + len(reaction_list):
        # no reduction was done, so this will be a full Monte Carlo sampling

        thermo_perturbations = np.random.multivariate_normal(mean=np.zeros(thermo_covariance_matrix.shape[0]), cov=thermo_covariance_matrix, size=CHUNK_SIZE) * 4184  # convert to J/mol
        kinetic_perturbations = np.random.multivariate_normal(mean=np.zeros(kinetic_covariance_matrix.shape[0]), cov=kinetic_covariance_matrix, size=CHUNK_SIZE)
        kinetic_multipliers = np.exp(kinetic_perturbations)
        kinetic_multipliers_ct = kinetic_multipliers.dot(ct2rmg_matrix.T)

        # save copies of all thermo for faster perturbation
        thermo_copies = []
        for sp_index in range(gas.n_species):
            thermo_copies.append(ct.Species().from_dict(gas.species()[sp_index].input_data.copy()))

        # Cantera does well if you give it lots of CPUs for a single simulation
        # but slows down if you try to parallelize different simulations across multiple processes
        # so we run the simulations in serial here do the parallelize across SLURM array jobs.
        for i in range(CHUNK_SIZE):

            # perturb all the species
            for sp_index in range(gas.n_species):
                # random perturbation
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
                gas.modify_species(sp_index, thermo_copies[sp_index])
            gas.set_multiplier(1.0)

        np.save(output_filename, y)

    else:
        raise NotImplementedError("Loading a problem description from a previous Morris screening is not implemented yet")


if __name__ == "__main__":
    settings_yaml = sys.argv[1]
    chunk_index = int(sys.argv[2])
    run_chunk(settings_yaml, chunk_index)
