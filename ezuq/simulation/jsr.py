"""Simulation code for a jet-stirred reactor"""
import cantera as ct
import numpy as np
import time


def run_simulation(gas, settings):
    """Run a JSR simulation with the given gas object and settings dictionary,
    which should contain temperature, pressure, composition, residence time, and volume"""

    T = settings['temperature']
    P = settings['pressure']
    X = settings['composition']
    RESIDENCE_TIME = settings['residence_time']
    VOLUME = settings['volume']

    gas.TPX = T, P, X

    r = ct.IdealGasReactor(gas, energy='off', name=f'JSR_{id(gas)}_{time.time()}')
    r.volume = VOLUME
    mass_flow_rate = r.mass / RESIDENCE_TIME

    upstream = ct.Reservoir(gas, name=f'upstream_{id(gas)}_{time.time()}')
    downstream = ct.Reservoir(gas, name=f'downstream_{id(gas)}_{time.time()}')

    m = ct.MassFlowController(upstream, r, mdot=mass_flow_rate)
    v = ct.PressureController(r, downstream, primary=m, K=1e-5)

    sim = ct.ReactorNet([r])
    sim.advance(1000.0 * RESIDENCE_TIME)

    concs = gas.X.copy()
    return concs


def run_simulation_with_sensitivity(gas, settings):
    """same as run_simulation but includes sensitivities
    These are kept as totally separate functions because run_simulations gets run A LOT for UQ
    and we don't want to waste overhead on figuring out if sensitivities are needed or not"""

    T = settings['temperature']
    P = settings['pressure']
    X = settings['composition']
    RESIDENCE_TIME = settings['residence_time']
    VOLUME = settings['volume']

    gas.TPX = T, P, X

    r = ct.IdealGasReactor(gas, energy='off', name=f'JSR_{id(gas)}_{time.time()}')
    r.volume = VOLUME
    mass_flow_rate = r.mass / RESIDENCE_TIME

    upstream = ct.Reservoir(gas, name=f'upstream_{id(gas)}_{time.time()}')
    downstream = ct.Reservoir(gas, name=f'downstream_{id(gas)}_{time.time()}')

    m = ct.MassFlowController(upstream, r, mdot=mass_flow_rate)
    v = ct.PressureController(r, downstream, primary=m, K=1e-5)

    sim = ct.ReactorNet([r])

    # add every species and reaction as a sensitivity species
    for j in range(gas.n_reactions):
        r.add_sensitivity_reaction(j)
    for j in range(gas.n_species):
        r.add_sensitivity_species_enthalpy(j)
    sensitivities = np.zeros((gas.n_reactions + gas.n_species, gas.n_species))

    sim.advance(1000.0 * RESIDENCE_TIME)

    concs = gas.X.copy()
    mass_fracs = gas.Y.copy()
    # record sensitivities
    for k in range(gas.n_species):
        for j in range(gas.n_reactions):  # gas reactions
            sensitivities[j, k] = sim.sensitivity(gas.species_names[k], j)
        for j in range(gas.n_species):  # gas species
            sensitivities[gas.n_reactions + j, k] = sim.sensitivity(gas.species_names[k], gas.n_reactions + j) * 4.184 * 1e6  # convert from J/kmol to kcal / mol in denominator    

    return concs, mass_fracs, sensitivities
