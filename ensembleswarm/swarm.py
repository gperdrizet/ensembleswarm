'''Creates and trains a swarm of level II regression ensembles.'''

import time
import pickle
import copy
from multiprocessing import Manager, Process, cpu_count
from pathlib import Path

import h5py
import numpy as np
# import pandas as pd
from sklearn.exceptions import ConvergenceWarning
# from sklearn.ensemble import HistGradientBoostingRegressor
import ensembleswarm.regressors as regressors


class Swarm:
    '''Class to hold ensemble model swarm.'''

    def __init__(
            self,
            ensembleset: str = 'ensembleset_data/dataset.h5',
            swarm_directory: str = 'ensembleswarm_models'
        ):

        # Check user argument types
        type_check = self.check_argument_types(
            ensembleset,
            swarm_directory
        )

        # If the type check passed, assign arguments to attributes
        if type_check is True:
            self.ensembleset = ensembleset
            self.swarm_directory = swarm_directory

        self.models = regressors.MODELS


    def train_swarm(self, sample: int = None) -> None:
        '''Trains an instance of each regressor type on each member of the ensembleset.'''

        Path(f'{self.swarm_directory}/swarm').mkdir(parents=True, exist_ok=True)

        manager=Manager()
        input_queue=manager.Queue(maxsize=5)

        swarm_trainer_processes=[]

        for i in range(cpu_count() - 2):
            print(f'Starting worker {i}')
            swarm_trainer_processes.append(
                Process(
                    target=self.train_model,
                    args=(input_queue,)
                )
            )

        for swarm_trainer_process in swarm_trainer_processes:
            swarm_trainer_process.start()

        with h5py.File(self.ensembleset, 'r') as hdf:
            num_datasets=len(list(hdf['train'].keys())) - 1
            print(f"Training datasets: {list(hdf['train'].keys())})")
            print(f'Have {num_datasets} sets of training features.')

            for swarm in range(num_datasets):

                Path(f'{self.swarm_directory}/swarm/{swarm}').mkdir(parents=True, exist_ok=True)

                features = hdf[f'train/{swarm}'][:]
                labels = hdf['train/labels'][:]
                models = copy.deepcopy(self.models)

                for model_name, model in models.items():

                    if sample is not None:
                        idx = np.random.randint(np.array(features).shape[0], size=sample)
                        features = features[idx, :]
                        labels = labels[idx]

                    work_unit = {
                        'swarm': swarm,
                        'model_name': model_name,
                        'model': model,
                        'features': features,
                        'labels': labels
                    }

                    input_queue.put(work_unit)

        for swarm_trainer_process in swarm_trainer_processes:
            input_queue.put({'swarm': 'Done'})

        for swarm_trainer_process in swarm_trainer_processes:
            swarm_trainer_process.join()
            swarm_trainer_process.close()

        manager.shutdown()


    def train_model(self, input_queue) -> None:
        '''Trains an individual swarm model.'''

        # Main loop
        while True:

            # Get next job from input
            work_unit = input_queue.get()

            # Unpack the workunit
            swarm = work_unit['swarm']

            if swarm == 'Done':
                return

            else:
                model_name = work_unit['model_name']
                model = work_unit['model']
                features = work_unit['features']
                labels = work_unit['labels']
                print(f'Training {model_name}, swarm {swarm}', end='\r')

                try:
                    if model_name == 'Gaussian Process' and features.shape[0] > 5000:
                        idx = np.random.randint(features.shape[0], size=5000)
                        features = features[idx, :]
                        labels = labels[idx]

                    _=model.fit(features, labels)

                except ConvergenceWarning:
                    print('\nCaught ConvergenceWarning while fitting '+
                          f'{model_name} in swarm {swarm}', end='')
                    model = None

                model_file=f"{model_name.lower().replace(' ', '_')}.pkl"

                with open(
                    f'{self.swarm_directory}/swarm/{swarm}/{model_file}',
                    'wb'
                ) as output_file:

                    pickle.dump(model, output_file)

            time.sleep(1)


    # def train_output_model(self):
    #     '''Trains model to make predictions based on swarm output.'''

    #     with h5py.File(self.ensembleset, 'r') as hdf:

    #         num_datasets=len(list(hdf['train'].keys())) - 1

    #         level_two_dataset={}

    #         for i in range(num_datasets):

    #             with open(f'{self.swarm_directory}/swarm/{i}.pkl', 'rb') as input_file:
    #                 models = pickle.load(input_file)

    #             for model_name, model in models.items():

    #                 if model is not None:

    #                     predictions = model.predict(np.array(hdf[f'test/{i}']))
    #                     level_two_dataset[f'{i}_{model_name}']=predictions.flatten()

    #         level_two_dataset['label'] = np.array(hdf['test/labels'])
    #         level_two_df = pd.DataFrame.from_dict(level_two_dataset)

    #         model = HistGradientBoostingRegressor()
    #         _ = model.fit(level_two_df.drop('label', axis=1), level_two_df['label'])

    #         with open(f'{self.swarm_directory}/output_model.pkl', 'wb') as output_file:
    #             pickle.dump(model, output_file)


    def check_argument_types(self,
            ensembleset: str,
            swarm_directory: str
    ) -> bool:

        '''Checks user argument types, returns true or false for all passing.'''

        check_pass = False

        if isinstance(ensembleset, str):
            check_pass = True

        else:
            raise TypeError('Ensembleset path is not a string.')

        if isinstance(swarm_directory, str):
            check_pass = True

        else:
            raise TypeError('Swarm directory path is not a string.')

        return check_pass
