import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from Cross_validation import cross_validate_with_params
from Learning_Rate_Distribution import learning_rate_distribution
from Objective_Value_Distribution import objective_value_distribution
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class BayesianOptimizerFnx:
    def __init__(
            self,
            objective_function,
            float_param_ranges={},
            int_param_candidates={},
            n_initial_points=10000,
            external_initial_points=None,
            max_iterations=1e4,
            no_new_converge=3,
            no_better_converge=10,
            kernel=RBF(),
            acquisition_type='PI',
            beta_lcb=0.5,
            epsilon=1e-7,
            n_samples=int(1e6),
            random_seed=None
    ):
        self.objective_function = objective_function
        self.float_param_ranges = float_param_ranges
        self.int_param_candidates = int_param_candidates
        self.max_iterations = int(max_iterations)
        self.no_new_converge = no_new_converge
        self.no_better_converge = no_better_converge
        self.acquisition_type = acquisition_type
        self.beta_lcb = beta_lcb
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.n_initial_points = n_initial_points
        self.random_seed = random_seed
        self.gpr = GPR(
            kernel=kernel,
            n_restarts_optimizer=50,
            random_state=self.random_seed
        )
        self.parse_param_names()
        self.get_ranges_and_candidates()
        self.initial_points = self.get_initial_points(external_initial_points)
        self.sampled_params = self.initial_points
        print('Evaluating initial points...')
        self.evaluated_scores = np.array(
            [self.objective_function(**self.check_int_params(dict(zip(self.param_names, p)))) for p in self.initial_points]
        )
        unique_indices = self.get_unique_indices(self.sampled_params)
        self.sampled_params = self.sampled_params[unique_indices]
        self.evaluated_scores = self.evaluated_scores[unique_indices]
        self.num_initial_points = len(self.sampled_params)
        self.gpr.fit(self.sampled_params, self.evaluated_scores)

    def parse_param_names(self):
        self.float_param_names = list(self.float_param_ranges.keys())
        self.int_param_names = list(self.int_param_candidates.keys())
        self.param_names = self.float_param_names + self.int_param_names

    def get_ranges_and_candidates(self):
        self.float_param_ranges_values = np.array(list(self.float_param_ranges.values()))
        self.int_param_candidates_values = list(self.int_param_candidates.values())

    def get_initial_points(self, external_initial_points):
        internal_initial_points = self.generate_random_params(self.n_initial_points)
        if external_initial_points is not None:
            num_values = np.array([len(choices) for choices in external_initial_points.values()])
            if not all(num_values == num_values[0]):
                raise Exception('Number of values for each parameter must be the same')
            if num_values.sum() != 0:
                points = []
                for param in self.param_names:
                    points.append(external_initial_points[param])
                points = np.array(points).T
                internal_initial_points = np.vstack((internal_initial_points, points))
        unique_indices = self.get_unique_indices(internal_initial_points)
        return internal_initial_points[unique_indices]

    def check_int_params(self, param_dict):
        for key, value in param_dict.items():
            if key in self.int_param_names:
                param_dict[key] = int(param_dict[key])
        return param_dict

    def generate_random_params(self, n):
        np.random.seed(self.random_seed)
        float_params = np.random.uniform(
            low=self.float_param_ranges_values[:, 0],
            high=self.float_param_ranges_values[:, 1],
            size=(int(n), self.float_param_ranges_values.shape[0])
        )
        if len(self.int_param_candidates) > 0:
            int_params = np.array([np.random.choice(choice, size=int(n)) for choice in self.int_param_candidates_values])
            int_params = int_params.T
            return np.hstack((float_params, int_params))
        else:
            return float_params

    def get_unique_indices(self, params):
        unique_params = np.unique(params, axis=0)
        if len(unique_params) == len(params):
            return list(range(len(params)))
        counter = {tuple(u): 0 for u in unique_params}
        indices = []
        for i, param in enumerate(params):
            param_tuple = tuple(param)
            if counter[param_tuple] == 0:
                counter[param_tuple] += 1
                indices.append(i)
        return indices

    def acquisition_function(self, params):
        print('Calculating acquisition values for sampled points based on GPR...')
        means, std_devs = self.gpr.predict(params, return_std=True)
        std_devs[std_devs < 0] = 0
        z = (self.evaluated_scores.min() - means) / (std_devs + self.epsilon)
        if self.acquisition_type == 'EI':
            return (self.evaluated_scores.min() - means) * norm.cdf(z) + std_devs * norm.pdf(z)
        if self.acquisition_type == 'PI':
            return norm.cdf(z)
        if self.acquisition_type == 'LCB':
            return means - self.beta_lcb * std_devs

    def min_acquisition(self, n=1e6):
        print('Performing random sampling based on parameter ranges and candidates...')
        params = self.generate_random_params(n)
        acquisition_values = self.acquisition_function(params)
        return params[acquisition_values.argmin()]

    def optimize(self):
        no_new_converge_counter = 0
        no_better_converge_counter = 0
        best_score = self.evaluated_scores.min()
        for i in range(self.max_iterations):
            print(f'Iteration: {i}, Current Best Score: {self.evaluated_scores.min()}')
            if no_new_converge_counter > self.no_new_converge:
                break
            if no_better_converge_counter > self.no_better_converge:
                break
            next_best_params = self.min_acquisition(self.n_samples)
            if np.any((self.sampled_params - next_best_params).sum(axis=1) == 0):
                no_new_converge_counter += 1
                continue
            print(f'Iteration {i}: Evaluating guessed best parameter set...')
            self.sampled_params = np.vstack((self.sampled_params, next_best_params))
            next_best_score = self.objective_function(**self.check_int_params(dict(zip(self.param_names, next_best_params))))
            self.evaluated_scores = np.append(self.evaluated_scores, next_best_score)
            print(f'Iteration {i}: Next Best Score: {next_best_score}, Parameters: {dict(zip(self.param_names, next_best_params))}')
            unique_indices = self.get_unique_indices(self.sampled_params)
            self.sampled_params = self.sampled_params[unique_indices]
            self.evaluated_scores = self.evaluated_scores[unique_indices]
            if self.evaluated_scores.min() < best_score:
                no_better_converge_counter = 0
                best_score = self.evaluated_scores.min()
            else:
                no_better_converge_counter += 1
            if len(self.sampled_params) == self.num_initial_points:
                no_new_converge_counter += 1
            else:
                no_new_converge_counter = 0
                self.num_initial_points = len(self.sampled_params)
            print(f'Iteration {i}: Re-fitting GPR with updated parameter sets...')
            self.gpr.fit(self.sampled_params, self.evaluated_scores)

    def get_results(self):
        num_initial = len(self.initial_points)
        num_new = len(self.evaluated_scores) - num_initial
        is_initial = np.array([1] * num_initial + [0] * num_new).reshape((-1, 1))
        results = pd.DataFrame(
            np.hstack((self.sampled_params, self.evaluated_scores.reshape((-1, 1)), is_initial)),
            columns=self.param_names + ['AvgTestCost', 'isInit']
        )
        return results.sort_values(by='AvgTestCost', inplace=False)


def bayesian_roc_auc(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)

    wine_data = load_wine()
    features, target = wine_data.data, wine_data.target
    
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    random_forest_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(random_forest_model, standardized_features, target, cv=kfold, scoring='roc_auc_ovr')

    return -np.mean(cv_scores)

def hyperopt_objective(params):
    n_estimators = int(params['n_estimators'])
    max_depth = int(params['max_depth'])
    min_samples_split = int(params['min_samples_split'])
    min_samples_leaf = int(params['min_samples_leaf'])

    data = load_wine()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc_ovr')

    # Negative mean ROC AUC (since we want to minimize the objective)
    return {'loss': -np.mean(scores), 'status': STATUS_OK}


float_param_ranges = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10)
}

int_param_candidates = {}

optimizer = BayesianOptimizerFnx(
    objective_function=bayesian_roc_auc,
    float_param_ranges=float_param_ranges,
    int_param_candidates=int_param_candidates,
    n_initial_points=100,
    max_iterations=100,
    acquisition_type='EI'
)

optimizer.optimize()

results = optimizer.get_results()
print(results)

best_params = results.iloc[0]
print("Best Parameters:")
print(best_params)

# Convergence Plot
plt.plot(results['AvgTestCost'], color='red')
plt.xlabel('Iteration')
plt.ylabel('Negative Mean ROC AUC')
plt.title('Convergence Plot')
plt.show()

print("------------------------------------------------------------------")
     
space = {
    'n_estimators': hp.quniform('n_estimators', 10, 2000, 1),
    'max_depth': hp.quniform('max_depth', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1)
}

trials = Trials()
best = fmin(
    fn=hyperopt_objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials,
    rstate=np.random.default_rng(42)
)

print("Best Parameters found by Hyperopt:")
print(best)

print("------------------------------------------------------------------")

bayesian_roc_auc_scores = results['AvgTestCost'].apply(lambda x: -x).values
hyperopt_roc_auc_scores = [-trial['result']['loss'] for trial in trials.trials]

# Bayesian versus Hyperopt Learning Rate Distribution
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.hist(bayesian_roc_auc_scores, bins=30, alpha=0.7, label='Bayesian Optimizer', color='red')
plt.xlabel('ROC AUC Score')
plt.ylabel('Frequency')
plt.title('Bayesian Optimizer Learning Rate Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(hyperopt_roc_auc_scores, bins=30, alpha=0.7, label='Hyperopt', color='black')
plt.xlabel('ROC AUC Score')
plt.ylabel('Frequency')
plt.title('Hyperopt Learning Rate Distribution')
plt.legend()

plt.tight_layout()
plt.show()

print("Best ROC AUC Score from Bayesian Optimizer: ", max(bayesian_roc_auc_scores))
print("Best ROC AUC Score from Hyperopt: ", max(hyperopt_roc_auc_scores))

print("------------------------------------------------------------------")

best_params_bayesian = results.iloc[0][['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']].to_dict()
bayesian_cv_scores = cross_validate_with_params(best_params_bayesian)

hyperopt_best_params = {
    'n_estimators': best['n_estimators'],
    'max_depth': best['max_depth'],
    'min_samples_split': best['min_samples_split'],
    'min_samples_leaf': best['min_samples_leaf']
}
hyperopt_cv_scores = cross_validate_with_params(hyperopt_best_params)

print("Cross-validation ROC AUC scores for Bayesian Optimizer best parameters: ", bayesian_cv_scores)
print("Cross-validation ROC AUC scores for Hyperopt best parameters: ", hyperopt_cv_scores)
print("Mean ROC AUC for Bayesian Optimizer: ", np.mean(bayesian_cv_scores))
print("Mean ROC AUC for Hyperopt: ", np.mean(hyperopt_cv_scores))

data = load_wine()
X, y = data.data, data.target
scaler = StandardScaler()
X = scaler.fit_transform(X)

# RandomForestClassifier
default_model = RandomForestClassifier(random_state=42)

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
default_scores = cross_val_score(default_model, X, y, cv=cv, scoring='roc_auc_ovr')

default_mean_auc = np.mean(default_scores)

print("Cross-validation ROC AUC scores for the default RandomForestClassifier:", default_scores)
print("Mean ROC AUC for the default RandomForestClassifier:", default_mean_auc)

bayesian_roc_auc_scores = bayesian_cv_scores
hyperopt_roc_auc_scores = hyperopt_cv_scores

learning_rate_distribution(bayesian_roc_auc_scores, hyperopt_roc_auc_scores, default_scores)

print("------------------------------------------------------------------")

bayesian_obj_values = results['AvgTestCost'].values
hyperopt_obj_values = [-trial['result']['loss'] for trial in trials.trials]

objective_value_distribution(bayesian_obj_values, hyperopt_obj_values)