import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class BayesianOptimizer:
    def __init__(
            self,
            func,
            float_param_ranges={},
            int_param_candidates={},
            n_init_points=10000,
            external_init_points=None,
            max_iter=1e4,
            no_new_converge=3,
            no_better_converge=10,
            kernel=RBF(),
            acq_type='PI',
            beta_lcb=0.5,
            eps=1e-7,
            n_sample=int(1e6),
            seed=None
    ):
        self.func = func
        self.float_param_dict = float_param_ranges
        self.int_param_dict = int_param_candidates
        self.max_iter = int(max_iter)
        self.no_new_converge = no_new_converge
        self.no_better_converge = no_better_converge
        self.acq_type = acq_type
        self.beta_LCB = beta_lcb
        self.eps = eps
        self.n_sample = n_sample
        self.n_init_points = n_init_points
        self.seed = seed
        self.gpr = GPR(
            kernel=kernel,
            n_restarts_optimizer=50,
            random_state=self.seed
        )
        self._parse_param_names()
        self._get_ranges_and_candidates()
        self.init_points = self._get_init_points(external_init_points)
        self.x = self.init_points
        print('Evaluating Initial Points...')
        self.y = np.array(
            [self.func(**self._check_int_param(dict(zip(self.param_names, p)))) for p in self.init_points]
        )
        u_index = self._unique_index(self.x)
        self.x = self.x[u_index]
        self.y = self.y[u_index]
        self.num_param_seeds = len(self.x)
        self.gpr.fit(self.x, self.y)

    def _parse_param_names(self):
        self.float_param_names = list(self.float_param_dict.keys())
        self.int_param_names = list(self.int_param_dict.keys())
        self.param_names = self.float_param_names + self.int_param_names

    def _get_ranges_and_candidates(self):
        self.float_param_ranges = np.array(list(self.float_param_dict.values()))
        self.int_param_candidates = list(self.int_param_dict.values())

    def _get_init_points(self, external_init_points):
        internal_init_points = self._generate_random_params(self.n_init_points)
        if external_init_points is not None:
            nums = np.array([len(choices) for choices in external_init_points.values()])
            if not all(nums == nums[0]):
                raise Exception('Number of values for each parameter must be the same')
            if nums.sum() != 0:
                points = []
                for param in self.param_names:
                    points.append(external_init_points[param])
                points = np.array(points).T
                internal_init_points = np.vstack((internal_init_points, points))
        u_index = self._unique_index(internal_init_points)
        return internal_init_points[u_index]

    def _check_int_param(self, param_dict):
        for k, v in param_dict.items():
            if k in self.int_param_names:
                param_dict[k] = int(param_dict[k])
        return param_dict

    def _generate_random_params(self, n):
        np.random.seed(self.seed)
        xs_range = np.random.uniform(
            low=self.float_param_ranges[:, 0],
            high=self.float_param_ranges[:, 1],
            size=(int(n), self.float_param_ranges.shape[0])
        )
        if len(self.int_param_dict) > 0:
            xs_candidates = np.array([np.random.choice(choice, size=int(n)) for choice in self.int_param_dict])
            xs_candidates = xs_candidates.T
            return np.hstack((xs_range, xs_candidates))
        else:
            return xs_range

    def _unique_index(self, xs):
        uniques = np.unique(xs, axis=0)
        if len(uniques) == len(xs):
            return list(range(len(xs)))
        counter = {tuple(u): 0 for u in uniques}
        indices = []
        for i, x in enumerate(xs):
            x_tuple = tuple(x)
            if counter[x_tuple] == 0:
                counter[x_tuple] += 1
                indices.append(i)
        return indices

    def _acquisition_func(self, xs):
        print('Calculating utility Acquisition on sampled points based on GPR...')
        means, sds = self.gpr.predict(xs, return_std=True)
        sds[sds < 0] = 0
        z = (self.y.min() - means) / (sds + self.eps)
        if self.acq_type == 'EI':
            return (self.y.min() - means) * norm.cdf(z) + sds * norm.pdf(z)
        if self.acq_type == 'PI':
            return norm.pdf(z)
        if self.acq_type == 'LCB':
            return means - self.beta_LCB * sds

    def _min_acquisition(self, n=1e6):
        print('Random sampling based on ranges and candidates...')
        xs = self._generate_random_params(n)
        ys = self._acquisition_func(xs)
        return xs[ys.argmin()]

    def optimize(self):
        no_new_converge_counter = 0
        no_better_converge_counter = 0
        best = self.y.min()
        for i in range(self.max_iter):
            print(f'Iteration: {i}, Current Best: {self.y.min()}')
            if no_new_converge_counter > self.no_new_converge:
                break
            if no_better_converge_counter > self.no_better_converge:
                break
            next_best_x = self._min_acquisition(self.n_sample)
            if np.any((self.x - next_best_x).sum(axis=1) == 0):
                no_new_converge_counter += 1
                continue
            print(f'Iteration {i}: evaluating guessed best param set by evaluation function...')
            self.x = np.vstack((self.x, next_best_x))
            next_best_y = self.func(**self._check_int_param(dict(zip(self.param_names, next_best_x))))
            self.y = np.append(self.y, next_best_y)
            print(f'Iteration {i}: next best is {next_best_y}, {dict(zip(self.param_names, next_best_x))}')
            u_index = self._unique_index(self.x)
            self.x = self.x[u_index]
            self.y = self.y[u_index]
            if self.y.min() < best:
                no_better_converge_counter = 0
                best = self.y.min()
            else:
                no_better_converge_counter += 1
            if len(self.x) == self.num_param_seeds:
                no_new_converge_counter += 1
            else:
                no_new_converge_counter = 0
                self.num_param_seeds = len(self.x)
            print(f'Iteration {i}: re-fit GPR with updated parameter sets')
            self.gpr.fit(self.x, self.y)

    def get_results(self):
        num_init = len(self.init_points)
        num_new = len(self.y) - num_init
        is_init = np.array([1] * num_init + [0] * num_new).reshape((-1, 1))
        results = pd.DataFrame(
            np.hstack((self.x, self.y.reshape((-1, 1)), is_init)),
            columns=self.param_names + ['AvgTestCost', 'isInit']
        )
        return results.sort_values(by='AvgTestCost', inplace=False)
