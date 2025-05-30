'''Unittests for Swarm class.'''

import os
import glob
import unittest
from shutil import rmtree

import pandas as pd
from sklearn.model_selection import train_test_split

import ensembleset.dataset as ds
from ensembleswarm.swarm import Swarm

class TestSwarm(unittest.TestCase):
    '''Tests for ensemble swarm class.'''

    def setUp(self):
        '''Dummy swarm instance for tests.'''

        # Ensembleset parameters
        self.n_datasets = 3
        self.frac_features = 0.1
        self.n_steps = 3

        self.ensembleset_directory = 'tests/ensemblesets'
        self.ensembleswarm_directory = 'tests/ensembleswarm_models'

        # Clear data directories
        if os.path.isdir(self.ensembleset_directory):
            rmtree(self.ensembleset_directory)

        # Clear data directories
        if os.path.isdir(self.ensembleswarm_directory):
            rmtree(self.ensembleswarm_directory)

        # Load and prep calorie data for testing
        data_df=pd.read_csv('tests/calories.csv')
        data_df=data_df.sample(frac=0.01)
        data_df.drop('id', axis=1, inplace=True, errors='ignore')
        train_df, test_df=train_test_split(data_df, test_size=0.5)
        train_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)

        # Set-up ensembleset
        self.dataset = ds.DataSet(
            label='Calories',
            train_data=train_df,
            test_data=test_df,
            string_features=['Sex'],
            data_directory=self.ensembleset_directory
        )

        # Generate datasets
        self.ensembleset_file = self.dataset.make_datasets(
            n_datasets=self.n_datasets,
            frac_features=self.frac_features,
            n_steps=self.n_steps
        )

        # Initialize ensembleswarm
        self.swarm = Swarm(
            ensembleset=f'{self.ensembleset_directory}/{self.ensembleset_file}',
            swarm_directory=self.ensembleswarm_directory,
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
        self.assertTrue(os.path.isdir(f'{self.ensembleswarm_directory}/swarm'))

        swarms=glob.glob(f'{self.ensembleswarm_directory}/swarm/*')
        self.assertEqual(len(swarms), self.n_datasets)


    def test_d_swarm_predict(self):
        '''Tests swarm prediction function.'''

        level_two_df, swarm_rmse_df = self.swarm.swarm_predict()

        self.assertTrue(isinstance(level_two_df, pd.DataFrame))
        self.assertTrue(isinstance(swarm_rmse_df, pd.DataFrame))
