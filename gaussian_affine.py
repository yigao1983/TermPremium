import numpy as np
import scipy.linalg as scipylin
import term_structure as ts


class GaussianAffine(ts.TermStructure):

    def __init__(self, **kwargs):
        super(GaussianAffine, self).__init__(**kwargs)

        lambda_a = self.params.lambda_a
        lambda_b = self.params.lambda_b
        sigma = self.params.sigma
        kappa = self.params.kappa
        theta = self.params.theta

        lambda_0 = np.matmul(sigma, lambda_a)
        lambda_x = np.matmul(sigma, lambda_b)

        kappa_rn = kappa + lambda_x
        theta_rn = np.dot(np.linalg.pinv(kappa_rn), np.dot(kappa, theta) - lambda_0)

        keig_rn, Xinv_rn = np.linalg.eig(kappa_rn)
        X_rn = np.linalg.pinv(Xinv_rn)
        Xsig_rn = np.dot(X_rn, sigma)
        Smat_rn = np.dot(Xsig_rn, Xsig_rn.T)

        self.params.setdefault('lambda_b', lambda_b)
        self.params.setdefault('kappa_rn', kappa_rn)
        self.params.setdefault('theta_rn', theta_rn)

        self.params.setdefault('keig_rn', keig_rn)
        self.params.setdefault('X_rn', X_rn)
        self.params.setdefault('Smat_rn', Smat_rn)

    def get_yield(self, tenor, **kwargs):

        b_t = self.get_bt(tenor)
        a_t = self.get_at(tenor)

        if 'x_t' not in kwargs:
            raise KeyError("No state vector x_t in kwargs")
        else:
            x_t = kwargs['x_t']
            yld = a_t + np.dot(b_t, x_t)
            return yld

    def get_bt(self, tenor):

        b0 = self.params.b0
        kappa_rn_T = self.params.kappa_rn.T
        b_t = np.dot(np.identity(len(b0)) - scipylin.expm(-kappa_rn_T * tenor),
                     np.dot(np.linalg.pinv(kappa_rn_T * tenor), b0))

        return b_t.real

    def get_at(self, tenor):

        Smat_rn = self.params.Smat_rn
        keig_rn = self.params.keig_rn

        coef = np.zeros(np.shape(Smat_rn))

        nrow, ncol = np.shape(coef)
        kernel = lambda k, t: (1. - np.exp(-k * t)) / (k * t)

        coef = np.array([[1. - kernel(keig_rn[i], tenor) - kernel(keig_rn[j], tenor) +
                          kernel(keig_rn[i] + keig_rn[j], tenor) for j in range(ncol)] for i in range(nrow)])

        Xi_t = np.multiply(Smat_rn, coef)

        a0 = self.params.a0
        b0 = self.params.b0
        theta_rn = self.params.theta_rn
        kappa_rn = self.params.kappa_rn
        X_rn = self.params.X_rn

        bkX = np.dot(b0, np.dot(np.linalg.pinv(kappa_rn), np.linalg.pinv(X_rn)))

        tr = np.dot(bkX, np.dot(Xi_t, bkX))

        a_t = a0 + np.dot(b0 - self.get_bt(tenor), theta_rn) - .5 * tr

        return a_t.real

    def get_expected_short(self, fwd_time, **kwargs):

        if 'x_t' not in kwargs:
            raise KeyError("No state vector x_t in kwargs")

        x_t = kwargs['x_t']
        a0 = self.params.a0
        b0 = self.params.b0
        theta = self.params.theta
        kappa = self.params.kappa

        exp_kappa_t = scipylin.expm(-kappa * fwd_time)

        z_t = np.dot(exp_kappa_t, x_t) + np.dot((np.identity(len(theta)) - exp_kappa_t), theta)

        return a0 + np.dot(b0, z_t)

    def get_expected_yield(self, fwd_time, tenor, **kwargs):

        if 'x_t' not in kwargs:
            raise KeyError("No state vector x_t in kwargs")

        a_t = self.get_at(tenor)
        b_t = self.get_bt(tenor)

        theta = self.params.theta
        kappa = self.params.kappa
        x_t = kwargs['x_t']

        exp_yld = a_t + \
                  np.dot(b_t, np.dot(np.identity(len(theta)) - scipylin.expm(-kappa * fwd_time), theta)) + \
                  np.dot(b_t, np.dot(scipylin.expm(-kappa * fwd_time), x_t))

        return exp_yld

    def get_fwd_rate(self, fwd_time, tenor, **kwargs):

        if 'x_t' not in kwargs:
            raise KeyError("No state vector x_t in kwargs")

        x_t = kwargs['x_t']

        yld_short = self.get_yield(fwd_time, x_t=x_t)
        yld_long = self.get_yield(fwd_time + tenor, x_t=x_t)

        fwd_rate = -(yld_short * fwd_time - yld_long * (fwd_time + tenor)) / tenor

        return fwd_rate

    def get_factor_cov(self, dt):

        ndim_x = len(self.params.theta)
        kappa = self.params.kappa
        sigma = self.params.sigma

        exp_Kh_sigma = np.dot(scipylin.expm(-kappa * dt), sigma)

        ker = np.dot(exp_Kh_sigma, exp_Kh_sigma.T) - np.dot(sigma, np.transpose(sigma))

        prefix = np.kron(kappa, np.identity(ndim_x)) + np.kron(np.identity(ndim_x), kappa)

        vec_cov = -np.dot(np.linalg.pinv(prefix), ker.flatten('F'))

        return np.reshape(vec_cov, (ndim_x, ndim_x))

    def get_term_premium(self, tenor_long, n_short, **kwargs):

        if 'x_t' not in kwargs:
            raise KeyError("No state vector x_t in kwargs")

        x_t = kwargs['x_t']

        tenor_short = float(tenor_long) / n_short

        yld_long = self.get_yield(tenor=tenor_long, x_t=x_t)

        yld_short_lst = [self.get_yield(tenor=tenor_short, x_t=x_t)] + \
                        [self.get_expected_yield(fwd_time=tenor_short * i_tenor, tenor=tenor_short, x_t=x_t)
                         for i_tenor in range(1, n_short)]

        return yld_long - np.average(yld_short_lst)


if __name__ == "__main__":
    dict_params_survey = {'theta': np.array([0., 0., 0.]),
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
                                                [-208.61290323, -162.43548387, 66.48387097]])}

    term_struct_obj = GaussianAffine(**dict_params_survey)
    term_struct_obj.update_param(sigma=dict_params_survey['sigma'])
    print("a_t = {}".format(term_struct_obj.get_at(1)))
    print("b_t = {}".format(term_struct_obj.get_bt(1)))
    print(term_struct_obj.get_factor_cov(1. / 252))
