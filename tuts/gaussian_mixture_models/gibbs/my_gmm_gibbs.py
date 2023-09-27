import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import  wishart, dirichlet, invwishart, multivariate_normal
from sklearn.metrics.cluster import adjusted_rand_score as ari
class GMMGibbsSampler():
    def __init__(self, X, K, burn_in=5, iterations=50, **kwargs):
        """
        [] input
        X = data n x d
        K = expected amount of components

        [] params = pi, z, mu, Sigma
        init z and pi using uniform alpha, mu and sigma using data and z assignments

        [] hyper-params = alpha, m0, V0, nu0, S0
        for uniform priors we set
        alpha0 -> small value
        m0 -> mean of init assignments
        V0 -> large eye matrix
        nu0 -> K + 1
        S0 -> data scatter matrix

        [] ss
        nk - count of data in each k
        x_bar - mean of data in each k
        """

        self.X = X  # data n x d
        self.K = K  # expected no of states
        self.burn_in = burn_in
        self.iterations = iterations
        self.N = X.shape[0] # length of data
        self.D = X.shape[1] # dimension of data

        self.Z_true = kwargs.get("Z_true")

        # set alpha to uniform prior and draw pi and z
        self.alpha0 = np.ones(K) * 1000/self.K # Uniform prior
        self.pi = dirichlet.rvs(alpha=self.alpha0, size=1).flatten()
        Z_mat = np.random.multinomial(n=1, pvals=self.pi, size=self.N) # N x K 'one-hot vectors'
        _, self.Z = np.where(Z_mat == 1) # N x 1 component number

        # sufficient stats
        self.nk = np.zeros(self.K, dtype=int)
        self.x_bar = np.zeros((self.K, self.D), dtype=float)
        for k in range(K):
            self.nk[k] = np.sum(self.Z == k)
            self.x_bar[k] = np.mean(self.X[self.Z == k], axis=0)

        # init mu and Sigma
        self.mu = np.zeros((self.K, self.D))
        self.sigma = np.zeros((self.K, self.D, self.D))
        self.lambdas = np.zeros((self.K, self.D, self.D))
        diagsig = np.zeros((self.K, self.D, self.D)) # scatter matrix
        for k in range(K):
            x_k = self.X[self.Z == k]
            sig_bar = np.cov(x_k.T, bias=True)
            diagsig[k] = np.diag(np.diag(sig_bar))
            self.mu[k] = self.x_bar[k]
            self.sigma[k] = sig_bar
            self.lambdas[k] = np.linalg.inv(sig_bar)

        # Hyper parameters
        # Mu
        self.m0 = self.x_bar # K x D
        self.V0 = [np.eye(self.D) * 1000 for _ in range(K)] # K x D x D
        # Sigma
        self.S0 = diagsig # K x D x D
        self.nu0 = self.K + 1  # 1 Degrees of freedom IW

        self.mu_trace = []
        self.sigma_trace = []
        self.pi_trace = []
        self.ARI = np.zeros((self.iterations))

    def fit(self):
        self.ARI = np.zeros((self.iterations))
        print('starting gibbs sampling')
        for it in range(self.iterations):
            self.gibbs_sweep()

            # save trace
            if it < self.burn_in:
                self.mu_trace.append(self.mu)
                self.sigma_trace.append(self.sigma)
                self.pi_trace.append(self.pi)

            # Calculate ARI
            if self.Z_true is not None:
                self.ARI[it] = np.round(ari(self.Z_true, self.Z), 3)
                print(f"ARI:{self.ARI[it]}")

        print('completed gibbs sampling')

    def get_model_likelihood(self):
        pass

    # one sweep of git sampler, return variables sampled
    def gibbs_sweep(self):
        # [1] sample params using assignments
        for k in range(self.K):
            # mu
            Vk = (np.linalg.inv(np.linalg.inv(self.V0[k]) + self.nk[k] * np.linalg.inv(self.sigma[k])))
            term1 = np.dot(np.linalg.inv(self.sigma[k]), self.nk[k] * self.x_bar[k])
            term2 = np.dot(np.linalg.inv(self.V0[k]), self.m0[k])
            mk = (np.dot(Vk, term1 + term2))
            # sample mu
            self.mu[k] = np.random.multivariate_normal(mean=mk, cov=Vk, size=1).flatten()

            # sigma
            dif = (self.X[self.Z == k] - self.mu[k])
            Sk = (self.S0[k] + (np.dot(dif.T, dif)))
            nuk = self.nu0 + self.nk[k]
            # sample sigma
            self.sigma[k] = invwishart.rvs(size=1, df=nuk, scale=Sk)

        # sample pi
        alpha_k = self.nk + self.alpha0
        self.pi = dirichlet.rvs(size=1, alpha=alpha_k).flatten()

        # [2] sample assignments using params
        # responsibilities
        res = np.zeros((self.N, self.K))
        for n in range(self.N):
            for k in range(self.K):
                res[n, k] = self.pi[k] * multivariate_normal.pdf(self.X[n], mean=self.mu[k], cov=self.sigma[k])
            res[n, :] /= np.sum(res[n, :])

        # Sample z
        Z_mat = np.zeros((self.N, self.K))
        for n in range(self.N):
            Z_mat[n] = np.random.multinomial(n=1, pvals=res[n], size=1).flatten()
        _, self.Z = np.where(Z_mat == 1)

        # [3] update ss
        for k in range(self.K):
            self.nk[k] = np.sum(self.Z == k)
            self.x_bar[k] = np.mean(self.X[self.Z == k], axis=0)

        return self.pi, self.mu, self.sigma

# my_gibbs = GMMGibbsSampler(x_i, 3, )
# np.sum(my_gibbs.Z == 0)