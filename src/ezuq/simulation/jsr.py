"""Simulation code for a jet-stirred reactor"""
import cantera as ct
import time


def run_simulation(gas, T_orig, P_orig, X_orig, VOLUME=9.5e-5, RESIDENCE_TIME=6.0):
    # https://pubs.acs.org/doi/10.1021/jp309821z residence time from Cord
    # https://www-sciencedirect-com.ezproxy.neu.edu/science/article/pii/S0010218022005703
    # VOLUME = 7.8e-5  # Zhu was 78 cm^3, Cord was 95cm^3  # it's a repeat of Cord et al

    T = T_orig
    P = P_orig
    X = X_orig

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
