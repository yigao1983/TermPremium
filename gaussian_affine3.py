import numpy as np
import term_structure as ts


class GaussianAffine3(ts.TermStructure):

    def __init__(self, **kwargs):
        super(GaussianAffine3, self).__init__(**kwargs)

        lambda_a = self.params['lambda_a']
        klambda_b = self.params['klambda_b']
        sigma = self.params['sigma']
        kappa = self.params['kappa']
        theta = self.params['theta']

        lambda_b = np.matmul(np.linalg.inv(kappa), klambda_b)

        lambda_0 = np.matmul(sigma, lambda_a)
        lambda_y = np.matmul(sigma, lambda_b)

        kappa_rn = kappa + lambda_y
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

    def get_yield(self, time2mat, **kwargs):

        b_t = self.get_bt(time2mat)
        a_t = self.get_at(time2mat)

        state_vec = kwargs['state_vec']

        yld = a_t + np.dot(b_t, state_vec)

        return yld

    def get_bt(self, time2mat):

        b0 = self.params['b0']
        kappa_rn_T = np.transpose(self.params['kappa_rn'])
        b_t = np.matmul(np.identity(len(b0)) - np.exp(-kappa_rn_T * time2mat),
                        np.matmul(np.linalg.inv(kappa_rn_T * time2mat), b0))

        return b_t

    def get_at(self, time2mat):

        Smat_rn = self.params['Smat_rn']
        keig_rn = self.params['keig_rn']

        coef = np.zeros(np.shape(Smat_rn))

        nrow, ncol = np.shape(coef)

        for irow in range(nrow):
            for icol in range(ncol):
                coef[irow, icol] = 1. - \
                                   (1. - np.exp(-keig_rn[irow] * time2mat)) / (keig_rn[irow] * time2mat) - \
                                   (1. - np.exp(-keig_rn[icol] * time2mat)) / (keig_rn[icol] * time2mat) + \
                                   (1. - np.exp(-(keig_rn[irow] + keig_rn[icol]) * time2mat)) / \
                                   ((keig_rn[irow] + keig_rn[icol]) * time2mat)

        Xi_t = np.multiply(Smat_rn, coef)

        a0 = self.params['a0']
        b0 = self.params['b0']
        theta_rn = self.params['theta_rn']
        kappa_rn = self.params['kappa_rn']
        X_rn = self.params['X_rn']

        bkX = np.matmul(b0, np.matmul(np.linalg.inv(kappa_rn), np.linalg.inv(X_rn)))

        tr = np.dot(bkX, np.matmul(Xi_t, bkX))

        a_t = a0 + np.dot(b0 - self.get_bt(time2mat), theta_rn) - .5 * tr

        return a_t

    def get_expected_yield(self, fwd_time, tenor, **kwargs):

        a_t = self.get_at(tenor)
        b_t = self.get_bt(tenor)

        theta = self.params['theta']
        kappa = self.params['kappa']
        state_vec = kwargs['state_vec']

        fwd_yld = a_t + \
                  np.dot(b_t, np.matmul(np.identity(len(theta)) - np.exp(-kappa * fwd_time), theta)) + \
                  np.dot(b_t, np.matmul(np.exp(-kappa * fwd_time), state_vec))

        return fwd_yld


if __name__ == "__main__":
    dict_params = {'theta': np.array([0., 0., 0.]),
                   'a0': 0.0486,
                   'b0': np.array([1., 1., 1.]),
                   'kappa': np.array([[0.0539, 0., 0.],
                                      [-0.1486, 0.4841, 0.],
                                      [0., -6.1308, 2.1521]]),
                   'sigma': np.array([[0.0023, 0., 0., ],
                                      [0., 0.0021, 0.],
                                      [0., 0., 0.0062]]),
                   'lambda_a': np.array([-0.7828, -0.6960, -1.8420]),
                   'klambda_b': np.array([[-0.1615, -0.8328, 0.2974],
                                          [-0.2564, -0.9081, 0.2889],
                                          [-1.2934, -1.0071, 0.4122]])}

    term_struct_obj = GaussianAffine3(**dict_params)

    term_struct_obj.get_yield(1.0)
