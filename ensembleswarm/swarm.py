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
import pandas as pd
from joblib import parallel_config
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.exceptions import ConvergenceWarning #, FitFailedWarning, UndefinedMetricWarning
import ensembleswarm.regressors as regressors


class Swarm:
    '''Class to hold ensemble model swarm.'''

    def __init__(
            self,
            ensembleset: str = 'ensembleset_data/dataset.h5',
            swarm_directory: str = 'ensembleswarm_models',
            model_types: list = None
        ):

        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

        # Check user argument types
        type_check = self.check_argument_types(
            ensembleset,
            swarm_directory,
            model_types
        )

        # If the type check passed, assign arguments to attributes
        if type_check is True:
            self.ensembleset = ensembleset
            self.swarm_directory = swarm_directory
            self.model_types = model_types

        self.models = {}
        self.hyperparameter_space = {}
        self.all_core_regressors = []

        if model_types is not None:
            for model_type in model_types:
                self.models[model_type] = regressors.MODELS[model_type]
                self.hyperparameter_space[model_type] = regressors.HYPERPARAMETERS[model_type]

                if model_type in regressors.ALL_CORE_REGRESSORS:
                    self.all_core_regressors.append(model_type)

        else:
            self.models = regressors.MODELS
            self.hyperparameter_space = regressors.HYPERPARAMETERS
            self.all_core_regressors = regressors.ALL_CORE_REGRESSORS


    def train_swarm(self, sample: int = None) -> None:
        '''Trains an instance of each regressor type on each member of the ensembleset.'''

        train_swarm_logger = logging.getLogger(__name__ + '.train_swarm')

        Path(f'{self.swarm_directory}/swarm').mkdir(parents=True, exist_ok=True)

        manager=Manager()
        input_queue=manager.Queue()

        swarm_trainer_processes=[]

        for i in range(1): #range(int(cpu_count() / 2)):
            train_swarm_logger.info('Starting worker %s', i)
            swarm_trainer_processes.append(
                Process(
                    target=self.train_model,
                    args=(input_queue,)
                )
            )

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

                    hyperparameters_file = (f'{self.swarm_directory}/swarm/{swarm}' +
                        f"/{model_name.replace(' ', '_').lower()}_hyperparameters.pkl")

                    hyperparameters = None

                    if Path(hyperparameters_file).is_file():
                        with open(hyperparameters_file, 'rb') as input_file:
                            hyperparameters = pickle.load(input_file)

                    if sample is not None:
                        idx = np.random.randint(np.array(features).shape[0], size=sample)
                        features = features[idx, :]
                        labels = labels[idx]

                    work_unit = {
                        'swarm': swarm,
                        'model_name': model_name,
                        'model': model,
                        'features': features,
                        'labels': labels,
                        'hyperparameters': hyperparameters
                    }

                    train_swarm_logger.info(
                        'Submitting ensemble %s, %s model for training',
                        swarm,
                        model_name
                    )
                    input_queue.put(work_unit)

        for swarm_trainer_process in swarm_trainer_processes:
            swarm_trainer_process.start()

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
                hyperparameters = work_unit['hyperparameters']

                if hyperparameters is not None:
                    model.set_params(**hyperparameters)
                    print(f'Training optimized {model_name}, swarm {swarm}', end='\r')

                elif hyperparameters is None:
                    print(f'Training {model_name}, swarm {swarm}', end='\r')

                try:

                    _=model.fit(features, labels)

                except ConvergenceWarning:
                    print('\nCaught ConvergenceWarning while fitting '+
                          f'{model_name} in swarm {swarm}')
                    model = None

                model_file=f"{model_name.lower().replace(' ', '_')}_model.pkl"

                with open(
                    f'{self.swarm_directory}/swarm/{swarm}/{model_file}',
                    'wb'
                ) as output_file:

                    pickle.dump(model, output_file)

            time.sleep(1)


    def optimize_swarm(
            self,
            sample: int = None,
            default_n_iter: int = 256,
            model_n_iter: dict = None,
            cv: int = 3
    ) -> None:
        '''Run per-model hyperparameter optimization using SciKit-learn's halving
        random search with cross-validation.'''

        optimize_swarm_logger = logging.getLogger(__name__ + '.optimize_swarm')

        Path(f'{self.swarm_directory}/swarm').mkdir(parents=True, exist_ok=True)

        results = {
            'ensemble': [],
            'model': [],
            'time': [],
            'score_mean': [],
            'score_std': []
        }

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

                    start_time = time.time()

                    time_thread = ElapsedTimeThread(model_name, ensemble, num_datasets)
                    time_thread.start()

                    if sample is not None:
                        idx = np.random.randint(np.array(features).shape[0], size=sample)
                        features = features[idx, :]
                        labels = labels[idx]

                    hyperparameters=self.hyperparameter_space[model_name]

                    n_iter = default_n_iter

                    if model_n_iter is not None:
                        if model_name in model_n_iter.keys():
                            n_iter = model_n_iter[model_name]

                    if n_iter is not None:

                        search_results = self.optimize_model(
                            ensemble,
                            model_name,
                            model,
                            features,
                            labels,
                            hyperparameters,
                            n_iter,
                            cv
                        )

                        if search_results is not None:

                            result = pd.DataFrame(search_results.cv_results_)
                            sorted_result = result.sort_values('rank_test_score')

                            results['model'].append(model_name)
                            results['ensemble'].append(ensemble)
                            results['score_mean'].append(
                                -sorted_result['mean_test_score'].to_list()[0]
                            )
                            results['score_std'].append(
                                sorted_result['std_test_score'].to_list()[0]
                            )
                            results['time'].append(time.time() - start_time)

                    time_thread.stop()
                    time_thread.join()

        results_df = pd.DataFrame.from_dict(results)
        results_df['efficiency_mean'] = results_df['score_mean'] / results_df['time']
        results_df['efficiency_std'] = results_df['score_std'] / results_df['time']

        return results_df


    def optimize_model(
            self,
            ensemble,
            model_name,
            model,
            features,
            labels,
            hyperparameters,
            n_iter,
            cv
    ) -> None:

        '''Optimizes an individual swarm model.'''

        optimize_model_logger = logging.getLogger(__name__ + '.optimize_model')

        n_jobs = cpu_count() - 2

        if model_name in self.all_core_regressors:
            n_jobs = 1

        hyperparameter_file=f"{model_name.lower().replace(' ', '_')}_hyperparameters.pkl"
        search_results_file=f"{model_name.lower().replace(' ', '_')}_optimization_results.pkl"

        if (
            Path(
                f'{self.swarm_directory}/swarm/{ensemble}/{hyperparameter_file}'
            ).is_file() and
            Path(
                f'{self.swarm_directory}/swarm/{ensemble}/{search_results_file}'
            ).is_file()
        ):

            optimize_model_logger.info('Already optimized %s, ensemble %s', model_name, ensemble)
            return None

        else:

            optimize_model_logger.info(
                'Optimizing %s, ensemble %s for %s iterations, with %s cv folds and n_jobs = %s',
                model_name,
                ensemble,
                n_iter,
                cv,
                n_jobs
            )

            try:

                search = RandomizedSearchCV(
                    model,
                    hyperparameters,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=n_jobs,
                    n_iter=n_iter,
                    cv=cv
                )

                with parallel_config(backend='multiprocessing'):
                    search.fit(features, labels)

                model = search.best_estimator_
                hyperparameters = search.best_params_

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
                f'{self.swarm_directory}/swarm/{ensemble}/{hyperparameter_file}',
                'wb'
            ) as output_file:

                pickle.dump(hyperparameters, output_file)

            with open(
                f'{self.swarm_directory}/swarm/{ensemble}/{search_results_file}',
                'wb'
            ) as output_file:

                pickle.dump(search, output_file)

            return search


    def swarm_predict(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''Run swarm prediction, returns level II dataset and individual model RMSE as two
        separate dataframes'''

        level_two_dataset = {}

        swarm_rmse = {
            'model': [],
            'ensemble': [],
            'RMSE': []
        }

        with h5py.File(self.ensembleset, 'r') as hdf:

            num_datasets=len(list(hdf['train'].keys())) - 1

            for i in range(num_datasets):

                for model_type in self.models:

                    time_thread = ElapsedTimeThread(model_type, i, num_datasets)
                    time_thread.start()

                    model_path = (f'{self.swarm_directory}/swarm/{i}/' +
                        f"{model_type.replace(' ', '_').lower()}_model.pkl")

                    with open(model_path, 'rb') as input_file:
                        model = pickle.load(input_file)

                    if model is not None and isinstance(model, dict) is False:

                        predictions = model.predict(hdf[f'test/{i}'][:])
                        level_two_dataset[f'{i}_{model_type}']=predictions.flatten()

                        rmse = root_mean_squared_error(hdf['test/labels'][:], predictions)

                        swarm_rmse['ensemble'].append(i)
                        swarm_rmse['model'].append(model_type)
                        swarm_rmse['RMSE'].append(rmse)

                    time_thread.stop()
                    time_thread.join()

            level_two_dataset['label'] = np.array(hdf['test/labels'])

        level_two_df = pd.DataFrame.from_dict(level_two_dataset)
        swarm_rmse_df = pd.DataFrame.from_dict(swarm_rmse)

        return level_two_df, swarm_rmse_df


    def check_argument_types(self,
            ensembleset: str,
            swarm_directory: str,
            model_types: list
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

        if isinstance(model_types, list) or model_types is None:
            check_pass = True

        else:
            raise TypeError('Model types is not a list.')

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

            update = str(f'\rRunning {self.model_name}, ensemble {self.ensemble + 1} ' +
                    f'of {self.num_datasets}, elapsed time: {elapsed_time:.0f} sec.')

            if elapsed_time >= 60 and elapsed_time < 3600:

                update = str(f'\rRunning {self.model_name}, ensemble {self.ensemble + 1} ' +
                    f'of {self.num_datasets}, elapsed time: {(elapsed_time / 60):.2f} min.')

            elif elapsed_time > 3600:

                update = str(f'\rRunning {self.model_name}, ensemble {self.ensemble + 1} ' +
                    f'of {self.num_datasets}, elapsed time: {(elapsed_time / 3600):.2f} hr.')

            print(update, end='')
            blank_len = len(update) + 10

            time.sleep(1)
