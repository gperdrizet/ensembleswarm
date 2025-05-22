'''Creates and trains a swarm of level II regression ensembles.'''

import threading
import logging
import time
import pickle
import copy
from multiprocessing import Manager, Process, cpu_count
from pathlib import Path

import h5py
import numpy as np
from joblib import parallel_config
from sklearn.experimental import enable_halving_search_cv # pylint: disable=W0611
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.exceptions import ConvergenceWarning #, FitFailedWarning, UndefinedMetricWarning
import ensembleswarm.regressors as regressors

# logging.captureWarnings(True)

class Swarm:
    '''Class to hold ensemble model swarm.'''

    def __init__(
            self,
            ensembleset: str = 'ensembleset_data/dataset.h5',
            swarm_directory: str = 'ensembleswarm_models'
        ):

        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

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
        self.hyperparameters = regressors.HYPERPARAMETERS


    def train_swarm(self, sample: int = None) -> None:
        '''Trains an instance of each regressor type on each member of the ensembleset.'''

        train_swarm_logger = logging.getLogger(__name__ + '.train_swarm')

        Path(f'{self.swarm_directory}/swarm').mkdir(parents=True, exist_ok=True)

        manager=Manager()
        input_queue=manager.Queue(maxsize=5)

        swarm_trainer_processes=[]

        for i in range(int(cpu_count() / 2)):
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
            train_swarm_logger.info('Training datasets: %s', list(hdf['train'].keys()))
            train_swarm_logger.info('Have %s sets of training features', num_datasets)

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
                          f'{model_name} in swarm {swarm}')
                    model = None

                model_file=f"{model_name.lower().replace(' ', '_')}.pkl"

                with open(
                    f'{self.swarm_directory}/swarm/{swarm}/{model_file}',
                    'wb'
                ) as output_file:

                    pickle.dump(model, output_file)

            time.sleep(1)


    def optimize_swarm(self, sample: int = None) -> None:
        '''Run per-model hyperparameter optimization using SciKit-learn's halving
        random search with cross-validation.'''

        optimize_swarm_logger = logging.getLogger(__name__ + '.optimize_swarm')

        Path(f'{self.swarm_directory}/swarm').mkdir(parents=True, exist_ok=True)

        with h5py.File(self.ensembleset, 'r') as hdf:
            num_datasets=len(list(hdf['train'].keys())) - 1
            optimize_swarm_logger.info('Training datasets: %s', list(hdf['train'].keys()))
            optimize_swarm_logger.info('Have %s sets of training features', num_datasets)

            for ensemble in range(num_datasets):

                Path(f'{self.swarm_directory}/swarm/{ensemble}').mkdir(parents=True, exist_ok=True)

                features = hdf[f'train/{ensemble}'][:]
                labels = hdf['train/labels'][:]
                models = copy.deepcopy(self.models)

                for model_name, model in models.items():

                    time_thread = ElapsedTimeThread(model_name, ensemble, num_datasets)
                    time_thread.start()

                    if sample is not None:
                        idx = np.random.randint(np.array(features).shape[0], size=sample)
                        features = features[idx, :]
                        labels = labels[idx]

                    hyperparameters=self.hyperparameters[model_name]

                    self.optimize_model(
                        ensemble,
                        model_name,
                        model,
                        features,
                        labels,
                        hyperparameters
                    )

                    time_thread.stop()
                    time_thread.join()


    def optimize_model(
            self,
            ensemble,
            model_name,
            model,
            features,
            labels,
            hyperparameters
    ) -> None:

        '''Optimizes an individual swarm model.'''

        optimize_model_logger = logging.getLogger(__name__ + '.optimize_model')

        model_file=f"{model_name.lower().replace(' ', '_')}.pkl"
        hyperparameter_file=f"{model_name.lower().replace(' ', '_')}_hyperparameters.pkl"

        if (
            Path(
                f'{self.swarm_directory}/swarm/{ensemble}/{model_file}'
            ).is_file() and
            Path(
                f'{self.swarm_directory}/swarm/{ensemble}/{hyperparameter_file}'
            ).is_file()
        ):

            optimize_model_logger.info('Already optimized %s, ensemble %s', model_name, ensemble)
            return

        else:

            optimize_model_logger.info('Optimizing %s, ensemble %s', model_name, ensemble)

            try:
                if model_name == 'Gaussian Process' and features.shape[0] > 5000:
                    idx = np.random.randint(features.shape[0], size=5000)
                    features = features[idx, :]
                    labels = labels[idx]

                search = HalvingRandomSearchCV(
                    model,
                    hyperparameters,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    cv=3
                )

                with parallel_config(backend='multiprocessing'):
                    search.fit(features, labels)

                model=search.best_estimator_
                hyperparameters=search.best_params_

            # except ConvergenceWarning as e:
            #     lines = str(e).splitlines()
            #     print('\nCaught ConvergenceWarning while fitting '+
            #         f'{model_name} in ensemble {ensemble}: {lines[1]}, {lines[-1]}', end='')
            #     model = None

            # except FitFailedWarning as e:
            #     lines = str(e).splitlines()
            #     print('\nCaught FitFailedWarning while optimizing '+
            #         f'{model_name} in ensemble {ensemble}: {lines[1]}, {lines[-1]}', end='')

            # except UndefinedMetricWarning as e:
            #     lines = str(e).splitlines()
            #     print('\nCaught UndefinedMetricWarning while optimizing '+
            #         f'{model_name} in ensemble {ensemble}: {lines[1]}, {lines[-1]}', end='')

            # except UserWarning as e:
            #     lines = str(e).splitlines()
            #     print('\nCaught UserWarning while optimizing '+
            #         f'{model_name} in ensemble {ensemble}: {lines[0]}', end='')

            except ValueError as e:
                lines = str(e).splitlines()
                optimize_model_logger.error(
                    'Caught ValueError while optimizing %s in ensemble %s: %s, %s',
                    model_name,
                    ensemble + 1,
                    lines[1], 
                    lines[-1]
                )

            with open(
                f'{self.swarm_directory}/swarm/{ensemble}/{model_file}',
                'wb'
            ) as output_file:

                pickle.dump(model, output_file)

            with open(
                f'{self.swarm_directory}/swarm/{ensemble}/{hyperparameter_file}',
                'wb'
            ) as output_file:

                pickle.dump(hyperparameters, output_file)


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


class ElapsedTimeThread(threading.Thread):
    '''Stoppable thread that prints the time elapsed'''

    def __init__(self, model_name, ensemble, num_datasets):
        super(ElapsedTimeThread, self).__init__()
        self._stop_event = threading.Event()
        self.model_name = model_name
        self.ensemble = ensemble
        self.num_datasets = num_datasets

    def stop(self):
        '''Stop method to stop timer printout.'''
        self._stop_event.set()

    def stopped(self):
        '''Method to check the timer state.'''
        return self._stop_event.is_set()

    def run(self):
        thread_start = time.time()

        blank_len = 90

        while not self.stopped():

            elapsed_time=time.time()-thread_start

            print(f'\r{" "*blank_len}', end='')

            if elapsed_time < 60:

                update = str(f'\rOptimizing {self.model_name}, ensemble {self.ensemble + 1} ' +
                    f'of {self.num_datasets}, elapsed time: {elapsed_time:.1f} sec.')

                print(update, end='')

            if elapsed_time >= 60 and elapsed_time < 3600:

                update = str(f'\rOptimizing {self.model_name}, ensemble {self.ensemble + 1} ' +
                    f'of {self.num_datasets}, elapsed time: {(elapsed_time / 60):.1f} min.')

                print(update, end='')

            if elapsed_time > 3600:

                update = str(f'\rOptimizing {self.model_name}, ensemble {self.ensemble + 1} ' +
                    f'of {self.num_datasets}, elapsed time: {(elapsed_time / 3600):.1f} hr.')

                print(update, end='')

            blank_len = len(update) + 10

            time.sleep(1)
