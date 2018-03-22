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
        theta_rn = np.matmul(np.linalg.inv(kappa_rn), np.matmul(kappa, theta) - lambda_0)

        keig_rn, Xinv_rn = np.linalg.eig(kappa_rn)
        X_rn = np.linalg.inv(Xinv_rn)
        Xsig_rn = np.matmul(X_rn, sigma)
        Smat_rn = np.matmul(Xsig_rn, np.transpose(Xsig_rn))

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

        b0 = self.params['b0']
        kappa_rn_T = np.transpose(self.params['kappa_rn'])
        b_t = np.matmul(np.identity(len(b0)) - scipylin.expm(-kappa_rn_T * tenor),
                        np.matmul(np.linalg.inv(kappa_rn_T * tenor), b0))

        return b_t

    def get_at(self, tenor):

        Smat_rn = self.params.Smat_rn
        keig_rn = self.params.keig_rn

        coef = np.zeros(np.shape(Smat_rn))

        nrow, ncol = np.shape(coef)
        kernel = lambda k, t: (1. - np.exp(-k * t)) / (k * t)

        coef = np.array([[1. - kernel(keig_rn[i], tenor) - kernel(keig_rn[j], tenor) +
                          kernel(keig_rn[i] + keig_rn[j], tenor) for i in range(nrow)] for j in range(ncol)])

        Xi_t = np.multiply(Smat_rn, coef)

        a0 = self.params.a0
        b0 = self.params.b0
        theta_rn = self.params.theta_rn
        kappa_rn = self.params.kappa_rn
        X_rn = self.params.X_rn

        bkX = np.matmul(b0, np.matmul(np.linalg.inv(kappa_rn), np.linalg.inv(X_rn)))

        tr = np.dot(bkX, np.matmul(Xi_t, bkX))

        a_t = a0 + np.dot(b0 - self.get_bt(tenor), theta_rn) - .5 * tr

        return a_t

    def get_expected_short(self, fwd_time, **kwargs):

        if 'x_t' not in kwargs:
            raise KeyError("No state vector x_t in kwargs")

        x_t = kwargs['x_t']
        a0 = self.params.a0
        b0 = self.params.b0
        theta = self.params.theta
        kappa = self.params.kappa

        exp_kappa_t = scipylin.expm(-kappa * fwd_time)

        z_t = np.matmul(exp_kappa_t, x_t) + np.matmul((np.identity(len(theta)) - exp_kappa_t), theta)

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
                  np.dot(b_t, np.matmul(np.identity(len(theta)) - scipylin.expm(-kappa * fwd_time), theta)) + \
                  np.dot(b_t, np.matmul(scipylin.expm(-kappa * fwd_time), x_t))

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

        exp_Kh_sigma = np.matmul(scipylin.expm(-kappa * dt), sigma)

        ker = np.matmul(exp_Kh_sigma, np.transpose(exp_Kh_sigma)) - np.matmul(sigma, np.transpose(sigma))

        prefix = np.kron(kappa, np.identity(ndim_x)) + np.kron(np.identity(ndim_x), kappa)

        vec_cov = -np.matmul(np.linalg.inv(prefix), ker.flatten('F'))

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
                          'lambda_b': np.array([[-0.1615, -0.8328, 0.2974],
                                                [-0.2564, -0.9081, 0.2889],
                                                [-1.2934, -1.0071, 0.4122]])}

    term_struct_obj = GaussianAffine(**dict_params_survey)
    term_struct_obj.update_param(sigma=dict_params_survey['sigma'])
    print("a_t = {}".format(term_struct_obj.get_at(1)))
    print("b_t = {}".format(term_struct_obj.get_bt(1)))
    print(term_struct_obj.get_factor_cov(1. / 252))
