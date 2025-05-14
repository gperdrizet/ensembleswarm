'''Creates and trains a swarm of level II regression ensembles.'''

import pickle
import copy
from pathlib import Path

import h5py
import numpy as np

import ensembleswarm.regressors as regressors
class Swarm:
    '''Class to hold ensemble model swarm.'''

    def __init__(
            self,
            ensembleset: str = 'data/dataset.h5'
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

        Path('data/swarm').mkdir(parents=True, exist_ok=True)

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

                with open(f'data/swarm/{i}.pkl', 'wb') as output_file:
                    pickle.dump(models, output_file)

