import os
import tempfile
import yaml
import subprocess
import pytest
import cantera as ct

import rmgpy.chemkin
import scipy.stats
import numpy as np


import ezuq.monte_carlo
import ezuq.morris_screen
import ezuq.util
import ezuq.simulation


@pytest.fixture(scope='class')
def tmp_working_dir():
    """Temporary directory that mimics the working_dir used in the demo."""
    with tempfile.TemporaryDirectory() as d:
        # make a symbolic link to the chem_annotated.yaml file in the examples directory
        ethane_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples', 'ethane')
        os.symlink(os.path.join(ethane_dir, 'chem_annotated.yaml'), os.path.join(d, 'chem_annotated.yaml'))
        os.symlink(os.path.join(ethane_dir, 'chem_annotated.inp'), os.path.join(d, 'chem_annotated.inp'))
        os.symlink(os.path.join(ethane_dir, 'species_dictionary.txt'), os.path.join(d, 'species_dictionary.txt'))
        os.symlink(os.path.join(ethane_dir, 'ct2rmg_rxn.pickle'), os.path.join(d, 'ct2rmg_rxn.pickle'))

        uncorr_ethane_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples', '00_uncorrelated_ethane')
        os.symlink(os.path.join(uncorr_ethane_dir, 'thermo_covariance_matrix.npy'), os.path.join(d, 'thermo_covariance_matrix.npy'))
        os.symlink(os.path.join(uncorr_ethane_dir, 'kinetic_covariance_matrix.npy'), os.path.join(d, 'kinetic_covariance_matrix.npy'))
        yield d

class TestMonteCarloUncorrelatedNoReduction:
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_working_dir):
        self.working_dir = tmp_working_dir

        # # make a symbolic link to the chem_annotated.yaml file in the examples directory
        # ethane_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples', 'ethane')
        # os.symlink(os.path.join(ethane_dir, 'chem_annotated.yaml'), os.path.join(self.working_dir, 'chem_annotated.yaml'))
        # os.symlink(os.path.join(ethane_dir, 'chem_annotated.inp'), os.path.join(self.working_dir, 'chem_annotated.inp'))
        # os.symlink(os.path.join(ethane_dir, 'species_dictionary.txt'), os.path.join(self.working_dir, 'species_dictionary.txt'))
        # os.symlink(os.path.join(ethane_dir, 'ct2rmg_rxn.pickle'), os.path.join(self.working_dir, 'ct2rmg_rxn.pickle'))

        # uncorr_ethane_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples', '00_uncorrelated_ethane')
        # os.symlink(os.path.join(uncorr_ethane_dir, 'thermo_covariance_matrix.npy'), os.path.join(self.working_dir, 'thermo_covariance_matrix.npy'))
        # os.symlink(os.path.join(uncorr_ethane_dir, 'kinetic_covariance_matrix.npy'), os.path.join(self.working_dir, 'kinetic_covariance_matrix.npy'))

        self.gas = ct.Solution(os.path.join(self.working_dir, 'chem_annotated.yaml'))

        # Specify conditions for the simulation
        temperatures = [800, 900, 1000]

        i_C2H4 = ezuq.util.get_i_thing_ct({'C': 2, 'H': 4}, self.gas)
        self.i_sens = i_C2H4
        i_C2H6 = ezuq.util.get_i_thing_ct({'C': 2, 'H': 6}, self.gas)
        i_O2 = ezuq.util.get_i_thing_ct({'O': 2}, self.gas)
        i_Ar = ezuq.util.get_i_thing_ct({'Ar': 1}, self.gas)  # He for Nancy data
        x_C2H6 = 0.18
        x_O2 = 0.64
        x_Ar = 1.0 - x_C2H6 - x_O2
        X = f'{self.gas.species_names[i_C2H6]}: {x_C2H6}, {self.gas.species_names[i_O2]}: {x_O2}, {self.gas.species_names[i_Ar]}: {x_Ar}'

        self.conditions = []
        for T in temperatures:
            self.conditions.append({
                'name': f'unreduced_{T}K',        # name for the directory where analysis will be run
                'temperature': T,       # K
                'pressure': ct.one_atm, # Pa
                'composition': X,
                'residence_time': 6.0,  # s
                'volume': 9.5e-5,       # m^3
            })

    def test_setup_runfiles(self):
        # Here we reduce the number of samples for much faster runtime, but you'll probably want to do ~100
        ezuq.monte_carlo.setup_runfiles(self.working_dir, self.conditions, morris_dir=None, i_sens=14)
        # check that the runfiles were created
        for condition in self.conditions:
            assert os.path.exists(os.path.join(self.working_dir, 'monte_carlo', condition['name']))
            assert os.path.exists(os.path.join(self.working_dir, 'monte_carlo', condition['name'], 'settings.yaml'))
        with open(os.path.join(self.working_dir, 'monte_carlo', self.conditions[0]['name'], 'settings.yaml'), 'r') as f:
            settings = yaml.safe_load(f)
        assert settings['temperature'] == self.conditions[0]['temperature']
        assert settings['pressure'] == self.conditions[0]['pressure']
        assert settings['composition'] == self.conditions[0]['composition']
        assert settings['residence_time'] == self.conditions[0]['residence_time']
        assert settings['volume'] == self.conditions[0]['volume']

    def test_run_monte_carlo(self):
        # Here we reduce the number of samples for much faster runtime
        ezuq.monte_carlo.setup_runfiles(self.working_dir, self.conditions, morris_dir=None, i_sens=14)
        condition = self.conditions[1]  # just run 900 K for testing

        condition_dir = os.path.join(self.working_dir, 'monte_carlo', condition['name'])
        my_settings_file = os.path.join(condition_dir, 'settings.yaml')
        for i in range(2):  # run 2000 samples for testing
            subprocess.check_call(['python', '-m', 'ezuq.monte_carlo', my_settings_file, str(i)])
        
        # check that the results were created
        condition_dir = os.path.join(self.working_dir, 'monte_carlo', condition['name'])
        assert os.path.exists(os.path.join(condition_dir, 'monte_carlo_y'))

        ezuq.monte_carlo.reassemble_chunks(condition_dir)

        assert os.path.exists(os.path.join(condition_dir, 'monte_carlo_results.npy'))
        results = np.load(os.path.join(condition_dir, 'monte_carlo_results.npy'))[:, self.i_sens]
        assert len(results) == ezuq.monte_carlo.CHUNK_SIZE * 2

        expected_median = 0.12038646245298754
        expected_std = 0.019559893706439853
        expected_2p5 = 0.07028704252084265
        expected_97p5 = 0.14523558479498513
        assert np.isclose(np.median(results), expected_median, rtol=0.1)
        assert np.isclose(np.std(results), expected_std, rtol=0.1)
        assert np.isclose(np.percentile(results, 2.5), expected_2p5, rtol=0.1)
        assert np.isclose(np.percentile(results, 97.5), expected_97p5, rtol=0.1)
