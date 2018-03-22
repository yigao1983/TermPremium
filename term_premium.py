import datetime as dt
import numpy as np
import scipy.linalg as scipylin
import pandas as pd
import matplotlib.pyplot as plt
import data_downloader as dd
import term_structure as ts
import gaussian_affine as ga
import pykalman as pk

NS1990 = ts.Parameters({'start_date': dt.date(1990, 1, 1),
                        'end_date': dt.date(2003, 12, 31),
                        'theta': np.array([0., 0., 0.]),
                        'a0': 0.0441,
                        'b0': np.array([1., 1., 1.]),
                        'kappa': np.array([[0.6360, 0., 0.],
                                           [5.8219, 0.5822, 0.],
                                           [4.5024, 0., 0.4985]]),
                        'sigma': np.array([[0.0022, 0., 0., ],
                                           [0., 0.0049, 0.],
                                           [0., 0., 0.0045]]),
                        'lambda_a': np.array([-0.5065, -1.9103, -1.6761]),
                        'slambda_b': np.array([[0.7744, -0.0459, 0.0912],
                                               [0., 0., -0.1052],
                                               [0, 0.31, -0.4572]]),
                        'lambda_b': np.array([[352., -20.86363636, 41.45454545],
                                              [0., 0., -21.46938776],
                                              [0., 68.88888889, -101.6]]),
                        'eta': np.diag([0.0010, 0.0003, 0.0005, 0.0000, 0.0003, 0.0000, 0.0005])})

S1990 = ts.Parameters({'start_date': dt.date(1990, 1, 1),
                       'end_date': dt.date(2003, 12, 31),
                       'theta': np.array([0., 0., 0.]),
                       'a0': 0.0486,
                       'b0': np.array([1., 1., 1.]),
                       'kappa': np.array([[0.0539, 0., 0.],
                                          [-0.1486, 0.4841, 0.],
                                          [0., -6.1308, 2.1521]]),
                       'sigma': np.array([[0.0023, 0., 0., ],
                                          [0., 0.0021, 0.],
                                          [0., 0., 0.0062]]),
                       'lambda_a': np.array([-0.7828, -0.6960, -1.8420]),
                       'lambda_b': np.array([[-70.2173913, -362.08695652, 129.30434783],
                                             [-122.0952381, -432.42857143, 137.57142857],
                                             [-208.61290323, -162.43548387, 66.48387097]]),
                       'eta': np.diag([0.0010, 0.0002, 0.0005, 0.0000, 0.0003, 0.0000, 0.0005])})


def calc_term_premium(tenor_lst, tenor_tp_lst, param_dict,
                      os_start, os_end):
    in_sample_df = dd.fetch_yld_data(tenor_lst, param_dict.start_date, param_dict.end_date)

    ga_obj = ga.GaussianAffine(**param_dict)

    exp_k_dt = scipylin.expm(-param_dict.kappa / 252.)
    F0 = np.matmul(np.identity(len(ga_obj.params.theta)) - exp_k_dt, ga_obj.params.theta)
    F1 = exp_k_dt
    H0 = np.array([ga_obj.get_at(t) for t in tenor_lst])
    H1 = np.array([ga_obj.get_bt(t) for t in tenor_lst])
    Q = ga_obj.get_factor_cov(1. / 252)
    R = param_dict.eta

    kf = pk.KalmanFilter(transition_matrices=F1, transition_offsets=F0, transition_covariance=Q,
                         observation_matrices=H1, observation_offsets=H0, observation_covariance=R)

    y = np.array(in_sample_df.values)

    kf = kf.em(y, n_iter=20, em_vars=['initial_state_mean', 'initial_state_covariance'])

    X, P = kf.smooth(y)

    model_obj = ga.GaussianAffine(**param_dict)

    for idx, (date, row) in enumerate(in_sample_df.iterrows()):
        x_t = X[idx, :]
        term_prem_dict = {t: model_obj.get_term_premium(tenor_long=t, n_short=12 * t, x_t=x_t)
                          for t in tenor_tp_lst}
        for t in tenor_tp_lst:
            in_sample_df.loc[date, 'TP_{}Y_1M'.format(t)] = term_prem_dict.get(t)

    in_sample_df[['TP_{}Y_1M'.format(t) for t in tenor_tp_lst]].plot()
    plt.savefig('term_perm_IS_1m.pdf')
    in_sample_df[['TP_{}Y_1M'.format(t) for t in tenor_tp_lst]].to_csv('term_prem_IS_1m.csv')

    out_sample_df = dd.fetch_yld_data(tenor_lst, os_start, os_end)

    kf_new = pk.KalmanFilter(transition_matrices=F1, transition_offsets=F0, transition_covariance=Q,
                             observation_matrices=H1, observation_offsets=H0, observation_covariance=R,
                             initial_state_mean=X[-1, :], initial_state_covariance=P[-1, :])

    X_os, P_os = kf_new.filter(out_sample_df.values)

    for idx, (date, row) in enumerate(out_sample_df.iterrows()):
        x_t = X_os[idx, :]
        term_prem_dict = {t: model_obj.get_term_premium(tenor_long=t, n_short=12 * t, x_t=x_t)
                          for t in tenor_tp_lst}
        for t in tenor_tp_lst:
            out_sample_df.loc[date, 'TP_{}Y_1M'.format(t)] = term_prem_dict.get(t)

    out_sample_df[['TP_{}Y_1M'.format(t) for t in tenor_tp_lst]].plot()
    plt.savefig('term_perm_OS_1m.pdf')
    out_sample_df[['TP_{}Y_1M'.format(t) for t in tenor_tp_lst]].to_csv('term_prem_OS_1m.csv')


if __name__ == '__main__':
    tenor_lst = [0.25, 0.5, 1, 2, 4, 7, 10]
    tenor_tp_lst = [1, 2, 5, 10]

    os_start = dt.date(2004, 1, 1)
    os_end = dt.date.today()

    calc_term_premium(tenor_lst, tenor_tp_lst, S1990, os_start, os_end)
