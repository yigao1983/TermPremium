import datetime as dt
import numpy as np
# import pandas as pd
import scipy.linalg as scipylin
import scipy.optimize as opt
import data_downloader as dd
import term_structure as ts
import gaussian_affine as ga
import pykalman as pk

# import kalman_filter as kf

RANDOM_SEED = 1
TENOR_LST = [0.25, 0.5, 1, 2, 4, 7, 10]


class KWCalibrator(object):
    n_runs = 0

    def __init__(self, sample_start, sample_end):
        self.__train_target = dd.fetch_yld_data(TENOR_LST, sample_start, sample_end).values

    @property
    def train_target(self):
        return self.__train_target

    def objective(self, x0):
        kappa11, kappa21, kappa22, \
        kappa31, kappa32, kappa33, \
        sigma11, sigma22, sigma33, \
        a0, \
        lambda_a1, lambda_a2, lambda_a3, \
        lambda_b11, lambda_b12, lambda_b13, \
        lambda_b21, lambda_b22, lambda_b23, \
        lambda_b31, lambda_b32, lambda_b33 = x0

        param_dict = ts.Parameters({
            'theta': np.zeros(3),
            'a0': a0,
            'b0': np.array([1, 1, 1]),
            'kappa': np.array([[kappa11, 0, 0],
                               [kappa21, kappa22, 0],
                               [kappa31, kappa32, kappa33]]),
            'sigma': np.diag([sigma11, sigma22, sigma33]),
            'lambda_a': np.array([lambda_a1, lambda_a2, lambda_a3]),
            'lambda_b': np.array([[lambda_b11, lambda_b12, lambda_b13],
                                  [lambda_b21, lambda_b22, lambda_b23],
                                  [lambda_b31, lambda_b32, lambda_b33]])
        })

        ga_obj = ga.GaussianAffine(**param_dict)

        exp_k_dt = scipylin.expm(-ga_obj.params.kappa / 252.)
        F0 = np.dot(np.identity(len(ga_obj.params.theta)) - exp_k_dt, ga_obj.params.theta)
        F1 = exp_k_dt
        H0 = np.array([ga_obj.get_at(t) for t in TENOR_LST])
        H1 = np.array([ga_obj.get_bt(t) for t in TENOR_LST])
        Q = ga_obj.get_factor_cov(1. / 252)

        kf_obj = pk.KalmanFilter(transition_offsets=F0, transition_matrices=F1, transition_covariance=Q,
                                 observation_offsets=H0, observation_matrices=H1, random_state=RANDOM_SEED)

        # kf_obj = kf.KalmanFilter(F0=F0, F1=F1, Q=Q, H0=H0, H1=H1, R=R)

        kf_obj.em(X=self.__train_target, em_vars=['initial_state_mean', 'initial_state_covariance',
                                                  'observation_covariance'])
        logll = kf_obj.loglikelihood(X=self.__train_target)

        self.n_runs += 1

        print(self.n_runs, logll / len(self.train_target))

        return -logll / len(self.train_target)

    def optimize(self, x0):
        # opt_result = opt.fmin(self.objective, x0, disp=True)
        opt_result = opt.minimize(self.objective, x0, method='Nelder-Mead', options={'disp': True})

        for k, v in opt_result.items():
            print('{}: {}'.format(k, v))

        return opt_result


if __name__ == '__main__':
    start_date = dt.date(1960, 1, 1)
    end_date = dt.date(1969, 12, 31)

    calib_obj = KWCalibrator(start_date, end_date)

    calib_obj.optimize([0.0539, -0.1486, 0.4841,
                        0, -6.1308, 2.1521,
                        0.0023, 0.0021, 0.0062,
                        0.0486,
                        -0.7828, -0.6960, -1.8420,
                        70.2173913, -362.08695652, 129.30434783,
                        -122.0952381, -432.42857143, 137.57142857,
                        -208.61290323, -162.43548387, 66.48387097])

    """
    sigma = np.array([[0.01, 0, 0],
                      [-0.007526, 0.01, 0],
                      [-0.044450, -0.009597, 0.01]])
    sigma_lambda = np.array([[-0.6953, 0.0339, -0.809],
                             [2.1331, -0.1447, 0.6],
                             [3.0734, -0.3576, 0.1553]])
    lambda_mat = np.matmul(np.linalg.inv(sigma), sigma_lambda)
    print(lambda_mat)

    calib_obj.optimize(np.array([0.8550, 0.1343, 1.4504,
                                 -0.007526, -0.044450, -0.009597,
                                 0.0474, 3.6695, 0.8844, 0.7169,
                                 0.3241, -0.4335, -1.2754,
                                 -69.53, 3.39, -80.9,
                                 160.981722, -11.918686, -0.88534,
                                 152.7733086, -32.12981295, -344.9201608]))
    """
