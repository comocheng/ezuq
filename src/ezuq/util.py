import cantera as ct


def get_i_thing_ct(ref_composition, phase):
    """Helper function for getting the index of a species in a Cantera phase given its composition"""
    for i in range(phase.n_species):
        if phase.species()[i].composition == ref_composition:
            return i
    assert False, f"Could not find species with composition {ref_composition} in phase {phase.name}"


def perturb_species(species, DELTA_J_MOL=418.4):
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
