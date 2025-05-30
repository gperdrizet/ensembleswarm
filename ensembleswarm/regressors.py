'''Regressor definitions and hyperparameter distributions for GridSearchCV with SciKit-learn.'''

from scipy.stats import uniform, loguniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor

MODELS={
    'Linear regression': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ]),
    'Quantile regression': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', QuantileRegressor())
    ]),
    'Nearest Neighbors': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', KNeighborsRegressor())
    ]),
    'Linear SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearSVR())
    ]),
    'RBF SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR(kernel='rbf'))
    ]),
    'Polynomial SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR(kernel='poly'))
    ]),
    'Gaussian Process': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GaussianProcessRegressor())
    ]),
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', DecisionTreeRegressor())
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor())
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', HistGradientBoostingRegressor())
    ]),
    'Neural Net': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MLPRegressor())
    ]),
    'AdaBoost': Pipeline(
        [('scaler', StandardScaler()),
         ('regressor', AdaBoostRegressor())
        ]),
    'SGD': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SGDRegressor())
    ]),
    # 'XGBoost': Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('regressor', XGBRegressor())
    # ]),
    # 'CatBoost': Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('regressor', CatBoostRegressor(silent=True))
    # ]),
    # 'LightGBM': Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('regressor', LGBMRegressor(verbosity=-1))
    # ])
}

HYPERPARAMETERS={
    'Linear regression': {
        'regressor__fit_intercept': [True, False],
        'regressor__positive': [True, False]
    },
    'Quantile regression': {
        'regressor__fit_intercept': [True, False],
        'regressor__alpha': loguniform(0.001, 10.0),
        'regressor__solver': ['highs-ds', 'highs-ipm', 'highs']
    },
    'Nearest Neighbors':{
        'regressor__n_neighbors': list(range(2, 4)),
        'regressor__weights': ['uniform', 'distance'],
        'regressor__leaf_size': list(range(5, 50)),
    },
    'Linear SVM':{
        'regressor__epsilon': loguniform(0.000001, 10.0),
        'regressor__C': loguniform(0.000001, 10.0),
        'regressor__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        'regressor__fit_intercept': [True, False],
        'regressor__intercept_scaling': uniform(loc=0, scale=10),
        'regressor__max_iter': list(range(500, 100000))
    },
    'RBF SVM':{
        'regressor__degree': list(range(1, 4)),
        # 'regressor__gamma': loguniform(0.001, 100.0),
        'regressor__C': loguniform(0.001, 10),
        'regressor__epsilon': loguniform(0.001, 10.0),
        'regressor__max_iter': list(range(100, 100000))
    },
    'Polynomial SVM':{
        'regressor__degree': list(range(1, 4)),
        # 'regressor__gamma': loguniform(0.001, 100.0),
        'regressor__coef0': loguniform(0.000001, 10.0),
        'regressor__C': loguniform(0.001, 10),
        'regressor__epsilon': loguniform(0.001, 10.0),
        'regressor__max_iter': list(range(100, 100000))
    },
    'Gaussian Process':{
        'regressor__n_restarts_optimizer': [0, 1, 2]
    },
    'Decision Tree':{
        'regressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'regressor__splitter': ['best', 'random'],
        'regressor__max_depth': [None, 10, 50, 100, 200],
        'regressor__min_samples_split': list(range(2, 20)),
        'regressor__min_samples_leaf': list(range(1, 100)),
        'regressor__min_weight_fraction_leaf': uniform(loc=0, scale=0.5),
        'regressor__max_features': loguniform(0.001, 1.0)
    },
    'Random Forest':{
        'regressor__n_estimators': list(range(10, 100)),
        'regressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'regressor__min_samples_split': list(range(2, 20)),
        'regressor__min_samples_leaf': list(range(1, 100)),
        'regressor__min_weight_fraction_leaf': uniform(loc=0, scale=0.5),
        'regressor__max_features': loguniform(0.001, 1.0),
        'regressor__min_impurity_decrease': loguniform(0.000001, 10.0),
        'regressor__ccp_alpha': loguniform(0.000001, 10.0)
    },
    'Gradient Boosting':{
        'regressor__loss': ['squared_error', 'absolute_error', 'gamma', 'poisson'],
        'regressor__learning_rate': loguniform(0.01, 1.0),
        'regressor__max_iter': list(range(50, 200)),
        'regressor__max_leaf_nodes': list(range(15, 60)),
        'regressor__min_samples_leaf': list(range(1, 100)),
        'regressor__l2_regularization': loguniform(0.000001, 1.0),
        'regressor__max_features': uniform(loc=0.25, scale=0.75),
        'regressor__interaction_cst': ['pairwise', 'no_interactions']
    },
    'Neural Net':{
        'regressor__hidden_layer_sizes': list(range(50, 200)),
        'regressor__solver': ['adam'], #,'lbfgs', 'sgd'],
        'regressor__alpha': loguniform(0.00001, 0.001),
        'regressor__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'regressor__learning_rate_init': loguniform(0.0001, 0.01),
        'regressor__max_iter': list(range(500, 1000))
    },
    'AdaBoost':{
        'regressor__n_estimators': list(range(5, 500)),
        'regressor__learning_rate': loguniform(0.001, 1.0)
    },
    'SGD':{
        'regressor__loss': [
            'squared_error',
            'huber', 
            'epsilon_insensitive',
            'squared_epsilon_insensitive'
        ],
        'regressor__penalty': ['l2', 'l1', 'elasticnet', None],
        'regressor__alpha': loguniform(0.00001, 0.1),
        'regressor__l1_ratio': loguniform(0.01, 1.0),
        'regressor__fit_intercept': [True, False],
        'regressor__max_iter': list(range(50000, 1000000)),
        'regressor__epsilon': loguniform(0.001, 1.0),
        'regressor__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'regressor__eta0': loguniform(0.00001, 0.1),
        'regressor__power_t': uniform(loc=-1, scale=1)
    },
    'XGBoost':{
        'regressor__n_estimators': list(range(5, 100)),
        'regressor__max_depth': list(range(1, 20)),
        'regressor__subsample': uniform(loc=0.1, scale=0.9)
    },
    'CatBoost':{
        'regressor__n_estimators': list(range(50, 1000)),
        'regressor__depth': list(range(1, 16)),
        'regressor__model_size_reg': loguniform(1e-9, 1e-5)
    },
    'LightGBM':{
        'regressor__learning_rate': loguniform(0.001, 1.0),
        'regressor__n_estimators': list(range(50, 500)),
        #'regressor__max_depth': list(range(1, 20)),
        'regressor__subsample': uniform(loc=0.1, scale=0.9)
    }
}

ALL_CORE_REGRESSORS = [
    'Linear regression',
    'Nearest Neighbors',
    'Gaussian Process',
    'Gradient Boosting', 
    'Neural Net'
]
