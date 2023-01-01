import numpy as np
# from scipy.fft import dct
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
plt.style.use('default')


class StatParams:
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov


class ParamMapper:
    def __init__(self, mu):
        self.dim = (mu.shape[1])
        self.states = mu.shape[0]
        self.km_sqr = np.power(self.states, 2)  # assume no more than two chains for the moment
        self.D = dct(np.eye(self.dim), type=2, axis=0, norm="ortho")
        self.D_inv = np.linalg.inv(self.D)

    def map_cepstral_to_log(self, stat_params: StatParams):
        mu_c = stat_params.mu
        cov_c = stat_params.cov
        # [1] cep => log
        mu_log = []
        cov_log = []
        for s in range(self.states):
            # mu_log.append(idct(x=mu_c[s], type=2, axis=0, norm="ortho"))
            mu_log.append(self.D_inv @ mu_c[s])
            cov_log.append(self.D_inv @ cov_c[s] @ self.D_inv.T)

        mu_log = np.array(mu_log)
        cov_log = np.array(cov_log)

        return StatParams(mu_log, cov_log)

    def map_log_to_linear(self, stat_params: StatParams):

        mu_log = stat_params.mu
        cov_log = stat_params.cov

        mu_lin = []
        cov_lin = []
        for s in range(self.states):
            mu_lin.append(np.exp(mu_log[s] + ((np.diag(cov_log[s, :, :])) / 2)))
        mu_lin = np.array(mu_lin)

        for s in range(self.states):
            temp_cov = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    temp_cov[i, j] = mu_lin[s, i] * mu_lin[s, j] * (np.exp(cov_log[s, i, j]) - 1)
            cov_lin.append(temp_cov)
        cov_lin = np.array(cov_lin)

        return StatParams(mu_lin, cov_lin)

    def map_cepstral_to_linear(self, params_cept: StatParams):
        # [1] cep => log
        params_log = self.map_cepstral_to_log(params_cept)

        # [2] log => lin
        params_lin = self.map_log_to_linear(params_log)

        return params_lin

    def map_log_to_cepstral(self, params_log: StatParams):

        mu_log = params_log.mu
        cov_log = params_log.cov

        # [1] log => cep
        x_mu_c_re = []
        x_cov_c_re = []
        for s in range(self.states):
            x_mu_c_re.append(self.D @ mu_log[s])
            x_cov_c_re.append(self.D @ cov_log[s] @ self.D.T)

        x_mu_c_re = np.array(x_mu_c_re)
        x_cov_c_re = np.array(x_cov_c_re)

        return StatParams(x_mu_c_re, x_cov_c_re)

    def map_linear_to_log(self, stat_params: StatParams):

        mu_lin = stat_params.mu
        cov_lin = stat_params.cov

        # [1] lin => log
        mu_log_re = []
        for s in range(self.states):
            mu_log_re.append(np.log(mu_lin[s]) - (0.5 * np.log(((np.diag(cov_lin[s])) / np.square(mu_lin[s])) + 1)))
        mu_log_re = np.array(mu_log_re)

        cov_log_re = []
        for s in range(self.states):
            temp_cov = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    temp_cov[i, j] = np.log((cov_lin[s, i, j] / (mu_lin[s, i] * mu_lin[s, j])) + 1)
            cov_log_re.append(temp_cov)
        cov_log_re = np.array(cov_log_re)

        return StatParams(mu_log_re, cov_log_re)

    def map_linear_to_cepstral(self, params_lin):

        # [1] lin => log
        params_log = self.map_linear_to_log(params_lin)

        # [2] log => cep
        params_cept = self.map_log_to_cepstral(params_log)

        return params_cept
