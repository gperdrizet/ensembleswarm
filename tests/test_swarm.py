'''Unittests for Swarm class.'''

import os
import glob
import logging
import unittest
from pathlib import Path
from shutil import rmtree

import pandas as pd
from sklearn.model_selection import train_test_split

import ensembleset.dataset as ds
from ensembleswarm.swarm import Swarm

Path('tests/logs').mkdir(parents=True, exist_ok=True)
logging.captureWarnings(True)
logger = logging.getLogger()

logging.basicConfig(
    filename='tests/logs/test_swarm.log',
    filemode='w',
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

ENSEMBLESET_DIRECTORY = 'tests/ensemblesets'
ENSEMBLESWARM_DIRECTORY = 'tests/ensembleswarm_models'

# Ensembleset parameters
N_DATASETS = 3
FRAC_FEATURES = 0.1
N_STEPS = 3

# Load and prep calorie data for testing
data_df = pd.read_csv('tests/calories.csv')
data_df = data_df.sample(frac=0.01)
data_df.drop('id', axis=1, inplace=True, errors='ignore')
data_df = data_df.sample(n=100)
train_df, test_df = train_test_split(data_df, test_size=0.5)
train_df.reset_index(inplace=True, drop=True)
test_df.reset_index(inplace=True, drop=True)

# Set-up ensembleset
dataset = ds.DataSet(
    label='Calories',
    train_data=train_df,
    test_data=test_df,
    string_features=['Sex'],
    data_directory=ENSEMBLESET_DIRECTORY
)

# Generate datasets
ENSEMBLESET_FILE = dataset.make_datasets(
    n_datasets=N_DATASETS,
    frac_features=FRAC_FEATURES,
    n_steps=N_STEPS
)


class TestSwarm(unittest.TestCase):
    '''Tests for ensemble swarm class.'''

    def setUp(self):
        '''Dummy swarm instance for tests.'''

        # Initialize ensembleswarm
        self.swarm = Swarm(
            ensembleset=f'{ENSEMBLESET_DIRECTORY}/{ENSEMBLESET_FILE}',
            swarm_directory=ENSEMBLESWARM_DIRECTORY,
            model_types=['Linear regression']
        )


    def test_a_class_arguments(self):
        '''Tests assignments of class attributes from user arguments.'''

        self.assertTrue(isinstance(self.swarm.ensembleset, str))
        self.assertTrue(isinstance(self.swarm.models, dict))

        with self.assertRaises(TypeError):
            _ = Swarm(ensembleset=0.0)


    def test_b_optimize_swarm(self):
        '''Tests ensembleswarm hyperparameter optimization.'''

        result_df = self.swarm.optimize_swarm(
            sample=100,
            default_n_iter=4,
            model_n_iter={'Neural Net': None}
        )

        self.assertTrue(isinstance(result_df, pd.DataFrame))


    def test_c_train_swarm(self):
        '''Tests fitting of ensemble swarm.'''

        self.swarm.train_swarm(sample = 100)
        self.assertTrue(os.path.isdir(f'{ENSEMBLESWARM_DIRECTORY }/swarm'))

        swarms=glob.glob(f'{ENSEMBLESWARM_DIRECTORY }/swarm/*')
        self.assertEqual(len(swarms), N_DATASETS)


    def test_d_swarm_predict(self):
        '''Tests swarm prediction function.'''

        level_two_df, swarm_rmse_df = self.swarm.swarm_predict()

        self.assertTrue(isinstance(level_two_df, pd.DataFrame))
        self.assertTrue(isinstance(swarm_rmse_df, pd.DataFrame))
