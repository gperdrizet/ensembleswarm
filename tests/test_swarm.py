'''Unittests for Swarm class.'''

import os
import glob
import pickle
import unittest
from pathlib import Path
from shutil import rmtree

import pandas as pd
from sklearn.model_selection import train_test_split

import ensembleset.dataset as ds
from ensembleswarm.swarm import Swarm

class TestSwarm(unittest.TestCase):
    '''Tests for ensemble swarm class.'''

    def setUp(self):
        '''Dummy swarm instance for tests.'''

        # Clear data directory
        if os.path.isdir('ensembleset_data'):
            rmtree('ensembleset_data')

        # Ensembleset parameters
        self.n_datasets = 3
        self.frac_features = 0.1
        self.n_steps = 3

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
            string_features=['Sex']
        )

        # Generate datasets
        self.dataset.make_datasets(
            n_datasets=self.n_datasets,
            frac_features=self.frac_features,
            n_steps=self.n_steps
        )

        # Initialize ensembleswarm
        self.swarm = Swarm()


    def test_class_arguments(self):
        '''Tests assignments of class attributes from user arguments.'''

        self.assertTrue(isinstance(self.swarm.ensembleset, str))
        self.assertTrue(isinstance(self.swarm.models, dict))

        with self.assertRaises(TypeError):
            _ = Swarm(ensembleset=0.0)


    def test_train_swarm(self):
        '''Tests fitting of ensemble swarm.'''

        self.swarm.train_swarm()

        self.assertTrue(os.path.isdir('ensembleset_data/swarm'))

        swarms=glob.glob('ensembleset_data/swarm/*pkl')
        self.assertEqual(len(swarms), self.n_datasets)

        with open(swarms[1], 'rb') as input_file:
            swarm = pickle.load(input_file)

        self.assertTrue(isinstance(swarm, dict))

        self.swarm.train_output_model()

        self.assertTrue(Path('ensembleset_data/swarm/output_model.pkl').is_file())
