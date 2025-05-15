'''Creates and trains a swarm of level II regression ensembles.'''

import pickle
import copy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import ensembleswarm.regressors as regressors
class Swarm:
    '''Class to hold ensemble model swarm.'''

    def __init__(
            self,
            ensembleset: str = 'ensembleset_data/dataset.h5'
        ):

        # Check user argument types
        type_check = self.check_argument_types(
            ensembleset
        )

        # If the type check passed, assign arguments to attributes
        if type_check is True:
            self.ensembleset = ensembleset

        self.models = regressors.MODELS


    def check_argument_types(self,
            ensembleset: str
    ) -> bool:

        '''Checks user argument types, returns true or false for all passing.'''

        check_pass = False

        if isinstance(ensembleset, str):
            check_pass = True

        else:
            raise TypeError('Ensembleset path is not a string.')

        return check_pass


    def train_swarm(self) -> None:
        '''Trains an instance of each regressor type on each member of the ensembleset.'''

        Path('ensembleset_data/swarm').mkdir(parents=True, exist_ok=True)

        with h5py.File(self.ensembleset, 'r') as hdf:

            num_datasets=len(list(hdf['train'].keys())) - 1

            for i in range(num_datasets):

                print(f'\nBuilding swarm {i+1} of {num_datasets}')
                models=copy.deepcopy(self.models)

                for model_name in models.keys():

                    print(f' Fitting {model_name}')

                    _=models[model_name].fit(
                        np.array(hdf[f'train/{i}']),
                        np.array(hdf['train/labels'])
                    )

                with open(f'ensembleset_data/swarm/{i}.pkl', 'wb') as output_file:
                    pickle.dump(models, output_file)


    def train_output_model(self):
        '''Trains model to make predictions based on swarm output.'''

        with h5py.File(self.ensembleset, 'r') as hdf:

            num_datasets=len(list(hdf['train'].keys())) - 1

            level_two_dataset={}

            for i in range(num_datasets):

                print(np.array(hdf[f'train/{i}']).shape)
                print(np.array(hdf[f'test/{i}']).shape)

                with open(f'ensembleset_data/swarm/{i}.pkl', 'rb') as input_file:
                    models = pickle.load(input_file)

                for model_name, model in models.items():

                    predictions = model.predict(np.array(hdf[f'test/{i}']))
                    level_two_dataset[f'{i}_{model_name}']=predictions.flatten()

            level_two_dataset['label'] = np.array(hdf['test/labels'])
            level_two_df = pd.DataFrame.from_dict(level_two_dataset)

            model = HistGradientBoostingRegressor()
            _ = model.fit(level_two_df.drop('label', axis=1), level_two_df['label'])

            with open('ensembleset_data/swarm/output_model.pkl', 'wb') as output_file:
                pickle.dump(model, output_file)
