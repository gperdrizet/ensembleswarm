'''Regressor definitions and hyperparameter distributions for GridSearchCV with SciKit-learn.'''

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
#from lightgbm import LGBMRegressor

REGRESSOR_JOBS = -1

MODELS={
    'Linear regression': LinearRegression(),
    'Quantile regression': QuantileRegressor(),
    'Nearest Neighbors':KNeighborsRegressor(n_jobs=REGRESSOR_JOBS),
    'Linear SVM':LinearSVR(max_iter=10000),
    'RBF SVM':SVR(kernel='rbf'),
    'Polynomial SVM':SVR(kernel='poly'),
    'Gaussian Process':GaussianProcessRegressor(),
    'Decision Tree':DecisionTreeRegressor(),
    'Random Forest':RandomForestRegressor(n_jobs=REGRESSOR_JOBS),
    'Neural Net':MLPRegressor(max_iter=1000),
    'AdaBoost':AdaBoostRegressor(),
    'SGD':SGDRegressor(penalty='elasticnet'),
    'XGBoost':XGBRegressor(n_jobs=REGRESSOR_JOBS),
    'CatBoost':CatBoostRegressor(thread_count=REGRESSOR_JOBS),
    #'LightGBM':LGBMRegressor(n_jobs=REGRESSOR_JOBS)
}

HYPERPARAMETERS={
    'Linear regression':{
        'fit_intercept':[True,False]
    },
    'Quantile regression':{
        'fit_intercept':[True,False],
        'alpha':[1.0]
    },
    'Nearest Neighbors':{
        'n_neighbors': [3, 4, 5],
        'weights': ['distance'],
        'algorithm': ['ball_tree'],
        'leaf_size': [30, 40, 50]
    },
    'Linear SVM':{
        'C': [1.0], 
        'epsilon': [0.0],
    },
    'RBF SVM':{
        'C': [0.24, 0.25, 0.26],
        'epsilon': [0.01, 0.1, 1]
    },
    'Polynomial SVM':{
        'C': [0.0125, 0.025, 0.05],
        'epsilon': [0.01, 0.1, 1],
        'degree': [2, 3],
        'coef0': [0.5, 1.0, 2]
    },
    'Gaussian Process':{
        'n_restarts_optimizer': [0]
    },
    'Decision Tree':{
        'criterion': ['squared_error'],
        'splitter': ['best'],
        'max_depth': [None, 100, 500],
        'max_features': [None, 0.5, 0.9]
    },
    'Random Forest':{
        'n_estimators': [300, 400, 500],
        'criterion': ['squared_error'],
        'max_depth': [None, 6, 7, 8],
        'max_features': [None, 0.75, 0.80, 0.85],
        'ccp_alpha': [0.0]
    },
    'Neural Net':{
        'hidden_layer_sizes': [3, 6, 12],
        'alpha': [0.2, 0.4, 0.8],
        'learning_rate': ['adaptive'],
        'max_iter': [1000]
    },
    'AdaBoost':{
        'n_estimators': [300, 400, 500],
        'learning_rate': [0.01, 0.02, 0.03],
        'loss': ['linear']
    },
    'SGD':{
        'loss': ['squared_error'],
        'alpha': [0.004, 0.005, 0.006],
        'l1_ratio': [0.55, 0.6, 0.65],
        'learning_rate': ['invscaling']
    },
    'XGBoost':{
        'n_estimators': [15, 20, 25],
        'max_depth': [2, 3, 4],
        'subsample': [1.0]
    },
    'CatBoost':{
        'n_estimators': [200, 300, 400],
        'depth': [1, 2, 3],
        'model_size_reg':[1e-9, 1e-8, 1e-7]
    },
    'LightGBM':{
        'learning_rate': [0.09, 0.1, 0.11],
        'n_estimators': [75, 100, 125],
        'max_depth': [1, 2, 3],
        'subsample': [0.25, 0.3, 0.35]
    }
}
