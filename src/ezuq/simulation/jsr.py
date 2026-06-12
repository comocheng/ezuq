"""Simulation code for a jet-stirred reactor"""
import cantera as ct
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
