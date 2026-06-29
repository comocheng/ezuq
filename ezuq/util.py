import os

import numpy as np
import yaml
import cantera as ct


def get_i_thing_rmg(thing, thing_list):
    for i in range(len(thing_list)):
        if thing.is_isomorphic(thing_list[i]):
            return i
    assert False


def get_i_thing_ct(ref_composition, phase):
    """Helper function for getting the index of a species in a Cantera phase given its composition"""
    for i in range(phase.n_species):
        if phase.species()[i].composition == ref_composition:
            return i
    assert False, f"Could not find species with composition {ref_composition} in phase {phase.name}"


def perturb_species_ct(species, DELTA_J_MOL=418.4):
    # takes in a Cantera species and makes a copy with the enthalpy offset changed
    # Default of 418 J/mol equals 0.1 kcal/mol
    R = 8.3144598  # gas constant in J/mol

    # copy the species
    input_data = species.input_data.copy()
    increase = None
    for i in range(len(input_data['thermo']['data'])):
        if not increase:
            # Only define the increase in enthalpy once or you'll end up with numerical gaps in continuity
            increase = DELTA_J_MOL / R
        input_data['thermo']['data'][i][5] += increase
    new_species = ct.Species().from_dict(input_data)
    return new_species


def setup_condition_dirs(base_dir, conditions):
    """You might want to run your simulation across temperatures, presssures, etc.
    This function helps you set up a directoty for each condition with a settings.yaml describing the simulation conditions.
    
    Conditions should be a list of dictionaries with info that will go in the settings yaml, along with the directory name. For example:
    conditions = [
        {temperature: 550, name: '550K'},
        {temperature: 650, name: '650K'},
        {temperature: 750, name: '750K'},
    ]
    """
    for condition in conditions:
        condition_dir = os.path.join(base_dir, condition['name'])
        os.makedirs(condition_dir, exist_ok=True)
        with open(os.path.join(condition_dir, 'settings.yaml'), 'w') as f:
            yaml.dump(condition, f, default_flow_style=False)


def is_diagonal(matrix):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    return np.all(matrix == np.diag(np.diagonal(matrix)))


def decompose_to_sqrt(matrix):
    # decompose a matrix into its square root
    # do cholesky decomposition if positive definite
    try:
        L = np.linalg.cholesky(matrix)
        assert np.isclose(L @ L.T, matrix).all()
        return L
    except np.linalg.LinAlgError as e:
        if 'Matrix is not positive definite' not in str(e):
            raise(e)

    # matrix is not positive semidefinite, so do an eigendecomposition instead and clip the negative/tiny eigenvalues to keep it invertible
    eig_values, eig_vectors = np.linalg.eigh(matrix)
    invertibility_floor = np.finfo(float).eps  # just enough to avoid divide-by-zero
    psd_floor = 0.0  # you decide: 0 if you trust the matrix is theoretically PSD

    eig_values_clipped = np.where(
        eig_values < psd_floor,
        invertibility_floor,   # replace negatives with tiny positive
        np.maximum(eig_values, invertibility_floor)  # leave positives alone (mostly)
    )

    # floor = eig_values.max() * len(eig_values) * np.finfo(float).eps  # scale the floor to the magnitude of the matrix values
    # eig_values_clipped = np.clip(eig_values, a_min=floor, a_max=None)
    L = eig_vectors @ np.diag(np.sqrt(eig_values_clipped))

    cov_reconstruction_error = eig_vectors @ np.diag(eig_values_clipped - eig_values) @ eig_vectors.T
    atol = np.abs(cov_reconstruction_error).max()
    #assert np.isclose(L @ L.T, matrix, rtol=1e-2, atol=np.maximum(atol, 1e-12)).all()
    return L
