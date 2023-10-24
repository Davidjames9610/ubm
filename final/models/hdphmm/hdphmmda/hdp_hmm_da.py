# Direct Assignment HDP-HMM
import time

import numpy as np
from hmmlearn.hmm import GaussianHMM

from final.models.hdphmm.hdphmmda.hdp_hmm_da_utils.hdp_hmm_da_consts import *

from sklearn.cluster import KMeans
from scipy.stats import dirichlet, beta
from sklearn.metrics.cluster import adjusted_rand_score as ari
from final.models.hdphmm.helpers.plot_hmm import plot_hmm_data
from numpy.random import binomial
from final.models.hdphmm.hdphmmda.hdp_hmm_da_utils.utils import *
from final.models.hdphmm.hdphmmwl.numba_wl import compute_probabilities, multinomial

class InfiniteDirectSamplerHMM():
    def __init__(self, X, K, Z_true, burn_in=0, iterations=20, verbose=False, sbp=None, temp=1):
        """
        Initializes the Infinite Direct Sampler Hidden Markov Model.

        Parameters:
            X (numpy.ndarray): Input data of shape (n x d).
            K (int): Expected number of states.
            Z_true (Optional): True labeling of X.
            burn_in (int): Gibbs iterations to burn before collecting samples.
            iterations (int): Number of iterations.

        Attributes:
            X (numpy.ndarray): Input data of shape (n x d).
            K (int): Current number of states.
            burn_in (int): Gibbs iterations to burn before collecting samples.
            iterations (int): Number of iterations.
            N (int): Length of data.
            D (int): Dimension of data.
            Z (None): Hidden states (to be initialized).
            Z_true: Optional true labeling of X.
            hmm (sklearn.hmm.GaussianHMM): Gaussian Hidden Markov Model to return
            tol (float): Tolerance for convergence.
            giw (dict): Gaussian-Inverse-Wishart parameters.
                - m0 (numpy.ndarray): Mean of init assignments.
                - k0 (int): Belief in m0.
                - nu0 (int): K + 1.
                - S0 (numpy.ndarray): Data scatter matrix.
            sbp (dict): Stick-Breaking Process variables and hyperparameters.
                - beta_vec (float):
                - beta_new (float):
                - alpha0 (float):
                - gamma0 (float):
                - kappa0: ...
                - rho0: ...
                - alpha0_a
                - alpha0_b

            aux (dict): Auxiliary variables.
                - m_mat: ...
                - m_init: ...
                - w_vec: ...
                - m_mat_bar: ...
            ss (dict): Sufficient statistics (to be updated).
                - nk: ...
                - n_mat: ...
                - n_ft: ...
            track (dict): Tracking variables for convergence and sampling progress.
                - ari: Array of zeros (length: iterations).
                - ll: List.
                - mu_trace: List.
                - sigma_trace: List.
                - a_trace: List.
                - pie_trace: List.
        """

        self.verbose = verbose
        self.X = X  # data n x d
        self.K = K  # current number of states
        self.burn_in = burn_in
        self.iterations = iterations
        self.N = X.shape[0]     # length of data
        self.D = X.shape[1]     # dimension of data
        self.Z = None
        self.Z_true = Z_true
        self.hmm = GaussianHMM(n_components=self.K, covariance_type="full")
        self.tol = 1e-2
        self.sbp_prior = sbp
        self.temp = temp
        self.tiny = 1e-300

        self.giw = {
            # M0: None,
            # K0: None,
            # NU0: None,
            # S0: None,
        }

        # SBP vars and hp
        self.sbp = {
            # BETA_VEC: None,
            # BETA_NEW: None,
            # ALPHA0: None,
            # GAMMA0: None,
            # KAPPA0: None,
            # RHO0: None,
        }

        self.aux = {
            # M_MAT: None,
            # M_INIT: None,
            # W_VEC: None,
            # M_MAT_BAR: None,
        }

        # Update this to include mean and sums at some point
        self.ss = {
            # NK: None,
            # N_MAT: None,
            # N_FT: None,
        }

        self.track = {
            ARI: np.zeros(self.iterations),
            LL: [],
            MU_TRACE: [],
            SIGMA_TRACE: [],
            A_TRACE: [],
            PIE_TRACE: []
        }

        self.init_sbp()
        self.init_z()
        self.init_giw_and_aux()
        self.update_ss(True)
        _ = self.get_hmm()


# [1] Init -------------------------------------------------------
    def init_sbp(self):

        # could also init with priors if sampling these too
        if self.sbp_prior:
            print('sbp_prior')
            self.sbp[GAMMA0] = self.sbp_prior[GAMMA0]
            self.sbp[KAPPA0] = self.sbp_prior[KAPPA0]
            self.sbp[ALPHA0] = self.sbp_prior[ALPHA0]
            self.sbp[RHO0]= self.sbp_prior[RHO0]
        else:
            self.sbp[GAMMA0] = 1           # 2.0 init gem
            self.sbp[KAPPA0] = 50          # 50 sticky-ness
            self.sbp[ALPHA0] = 2         # 50 concentration for 2nd stick breaking
            self.sbp[RHO0]= self.sbp[KAPPA0]/(self.sbp[KAPPA0]+self.sbp[ALPHA0]) # 0.5


        # inits beta vec
        beta_vec = dirichlet.rvs(np.array([1, self.sbp[GAMMA0]]), size=1)[0]
        beta_new = beta_vec[-1]
        beta_vec = beta_vec[:-1]

        for k in range(self.K - 1):
            b = beta.rvs(1, self.sbp[GAMMA0], size=1)
            beta_vec = np.hstack((beta_vec, b*beta_new))
            beta_new = (1-b)*beta_new

        self.sbp[BETA_VEC] = beta_vec
        self.sbp[BETA_NEW] = beta_new

    def init_z(self):

        self.Z = np.zeros(self.N)

        # true means
        kmeans = KMeans(n_clusters=self.K, random_state=42, init='random')
        kmeans.fit(self.X)

        # shuffle labels
        num_labels_to_replace = int(0.5 * len(kmeans.labels_))
        # Generate random labels between 0 and k
        random_labels = np.random.randint(0, self.K, num_labels_to_replace)
        # Replace 10% of the labels with random numbers
        shuffled_labels = np.copy(kmeans.labels_)
        replace_indices = np.random.choice(len(shuffled_labels), num_labels_to_replace, replace=False)
        shuffled_labels[replace_indices] = random_labels
        # Assign the shuffled labels to self.Z
        self.Z = shuffled_labels
        if self.Z_true is not None:
            print('init ari', np.round(ari(self.Z_true, self.Z), 3))

    def init_giw_and_aux(self):

        sig_bar = np.cov(self.X.T, bias=True)
        diagsig = np.diag(np.diag(sig_bar))
        self.giw[M0] = np.mean(self.X, axis = 0)
        self.giw[K0] = 0.1
        self.giw[S0] = np.eye(self.D, self.D) * 10 # diagsig   #
        self.giw[NU0] = np.copy(self.D) + 2  # 1 Degrees of freedom IW, nu0 + 2 will mean <sigma> = S0

        self.aux[M_MAT] = np.zeros((self.K,self.K))
        self.aux[M_MAT_BAR] = np.zeros((self.K,self.K))
        self.aux[W_VEC] = np.zeros(self.K)
        self.aux[M_INIT] = None

# [2] Gibbs sampling -------------------------------------------------------

# [2.a] Sampling Z ---------------------------------------------------------
    # predictive prob of x belonging to params of NIW
    def student_t_post(self, x):
        k0 = self.giw[K0]
        s0 = np.copy(self.giw[S0])
        m0 = np.copy(self.giw[M0])
        nu0 = self.giw[NU0]

        Sc = np.diag(np.diag(s0 + (np.outer(m0, m0) * k0)))

        outer_c = np.zeros((self.N, self.D, self.D))
        for i in range(self.N):
            outer_c[i, :, :] = np.outer(self.X[i], self.X[i])

        index_of_x = np.where(self.X == x)[0][0]
        class_of_x = self.Z[index_of_x]

        probs = np.zeros(self.K)
        for k in range(self.K):
            if self.ss[NK][k] > 0:
                if(class_of_x == k):
                    print('check')
                # a lot of this should be cached
                kn = k0 + self.ss[NK][k]
                nun = nu0 + self.ss[NK][k]
                mc = (k0 * m0) / kn

                # calculate Sx
                Sx = np.sum(outer_c[self.Z == k], axis=0) # -

                # mn
                # mn_top_a = (self.nk[k] * self.x_bar[k])
                mn_top_b = np.sum(self.X[self.Z == k], axis = 0)
                mn = mc + mn_top_b / kn

                # Sn
                Sn = Sc + np.diag(np.diag(Sx)) - np.diag(np.diag(np.outer(mn, mn) * kn))

                if(np.all(Sn >= 0)):
                    prob_k = student_t_giw(x, mn, kn, nun, Sn, self.D)
                else:
                    index_of_x = np.where(self.X == x)[0][0]
                    class_of_x = self.Z[index_of_x]
                    prob_k = student_t_giw(x, mn, kn, nun, Sc, self.D)
                probs[k] = prob_k
            else:
                probs[k] = student_t_giw(x, self.giw[M0], self.giw[K0], self.giw[NU0], self.giw[S0], self.D)

        x_dist_new = student_t_giw(x, self.giw[M0], self.giw[K0], self.giw[NU0], self.giw[S0], self.D)
        return probs, x_dist_new

    def get_crf_prob(self, t):

        # define vars for easy reading and avoid changing
        alpha0 = self.sbp[ALPHA0]
        kappa0 = self.sbp[KAPPA0]
        beta_vec = np.copy(self.sbp[BETA_VEC])
        beta_new = self.sbp[BETA_NEW]
        n_mat = np.copy(self.ss[N_MAT])

        tmp_vec = np.arange(self.K)
        j = self.Z[t-1]

        if t < self.N - 1:

            l = self.Z[t+1]

            zt_dist = \
                (alpha0*beta_vec + n_mat[j]+kappa0*(j==tmp_vec))\
                /(alpha0 + n_mat[j].sum()+kappa0)

            ztplus1_dist = \
                (alpha0*beta_vec[l] + n_mat[:,l] + kappa0*(l == tmp_vec) + (j==l)*(j == tmp_vec))\
                /(alpha0 + n_mat.sum(axis=1) + kappa0+(j == tmp_vec))

            new_dist = \
                (alpha0**2)*beta_vec[l]*beta_new\
                /((alpha0+kappa0)*(alpha0+n_mat[j].sum()+kappa0))

            return_value = np.log((zt_dist*ztplus1_dist) + self.tiny), np.log(new_dist + self.tiny)
            return return_value
        else:

            zt_dist = \
                (alpha0*beta_vec + n_mat[j]+kappa0*(j==tmp_vec))\
                /(alpha0 +n_mat[j].sum()+kappa0)

            new_dist = \
                alpha0*beta_new\
                /(alpha0+n_mat[j].sum()+kappa0)

            return np.log(zt_dist + self.tiny), np.log(new_dist + self.tiny)
    @staticmethod
    def boltzmann_softmax(logits, temperature=1.0, small_amount=1e-5):
        exp_logits = np.exp(logits / temperature) + small_amount
        probabilities = exp_logits / np.sum(exp_logits)
        return probabilities

    def sample_z(self):

        for t in range(1, self.N):
            # remove current assignment of z_t from stats and counts

            x = self.X[t]
            self.remove_x_from_z(t, x)

            # log prob of CRF
            zt_dist, zt_dist_new = self.get_crf_prob(t)

            # log prob of x | hyper-params
            x_dist, x_dist_new = self.student_t_post(x)

            # sample z
            post_cases = np.hstack((zt_dist+x_dist, zt_dist_new+x_dist_new))

            post_cases_probs = compute_probabilities(post_cases)
            self.Z[t] = multinomial(post_cases_probs)

            # new state
            if self.Z[t] == self.K:
                print('new state')
                # sampled beta
                b = beta.rvs(1, self.sbp[GAMMA0], size=1)
                self.sbp[BETA_VEC] = np.hstack((self.sbp[BETA_VEC], b*self.sbp[BETA_NEW]))
                self.sbp[BETA_NEW] = (1-b)*self.sbp[BETA_NEW]

                # update ss
                self.ss[N_MAT] = np.hstack((self.ss[N_MAT], np.zeros((self.K,1))))
                self.ss[N_MAT] = np.vstack((self.ss[N_MAT], np.zeros((1,self.K+1))))
                self.ss[N_FT] = np.hstack((self.ss[N_FT], [0]))
                self.ss[NK] = np.hstack((self.ss[NK], [0]))
                self.K += 1

            # update ss
            self.add_x_to_z(t, x)
            self.handle_empty_components()

        self.assign_first_point()

    # re-assign first point to same as second for the moment
    def assign_first_point(self):

        zt = self.Z[0]
        l = self.Z[1]
        self.ss[NK][zt] -= 1
        self.ss[N_MAT][zt, l] -= 1

        self.Z[0] = l
        self.ss[NK][l] += 1
        self.ss[N_MAT][l, l] += 1

    # remove x from z and update ss
    def remove_x_from_z(self, t, x):
        # t is current index of z
        # x is data
        zt = self.Z[t]
        j = self.Z[t-1]

        self.ss[NK][zt] -= 1
        self.ss[N_MAT][j, zt] -=1
        if t + 1 < self.N:
            l = self.Z[t+1]
            self.ss[N_MAT][zt, l] -=1

        self.Z[t] = -1  # will be re-assigned above

    def add_x_to_z(self, t, x):
        # t is current index of z
        # x is data
        zt = self.Z[t]
        j = self.Z[t-1]

        self.ss[NK][zt] += 1
        self.ss[N_MAT][j, zt] +=1
        if t + 1 < self.N:
            l = self.Z[t+1]
            self.ss[N_MAT][zt, l] +=1

    def handle_empty_components(self):
        nk = np.zeros(self.K)
        for k in range(self.K):
            nk[k] = np.sum(self.Z == k)
        zero_indices = np.where(nk == 0)[0]
        if len(zero_indices) > 0:
            print('deleting component(s), ', zero_indices)
            if len(zero_indices) > 1:
                print('more than 1 zero cluster')
            rem_ind = np.unique(self.Z)
            d = {k: v for v, k in enumerate(sorted(rem_ind))}
            self.Z = np.array([d[x] for x in self.Z])
            self.ss[N_MAT] = self.ss[N_MAT][rem_ind][:,rem_ind]
            self.ss[N_FT] = self.ss[N_FT][rem_ind]
            self.ss[NK] = self.ss[NK][rem_ind]
            self.sbp[BETA_VEC] = self.sbp[BETA_VEC][rem_ind]
            self.K = len(rem_ind)
            self.update_ss()

# [2.b] Sampling others
    def sample_aux_vars(self):
        m_mat = np.zeros((self.K,self.K))

        for j in range(self.K):
            for k in range(self.K):
                if self.ss[N_MAT][j,k] == 0:
                    m_mat[j,k] = 0
                else:
                    # pretend multivariate to len(n_mat) at once and avoid loop
                    x_vec = binomial(1, (self.sbp[ALPHA0]*self.sbp[BETA_VEC][k]+self.sbp[KAPPA0]*(j==k))
                                     /(np.arange(self.ss[N_MAT][j,k])+self.sbp[ALPHA0]*self.sbp[BETA_VEC][k]+self.sbp[KAPPA0]*(j==k)))
                    x_vec = np.array(x_vec).reshape(-1)
                    m_mat[j,k] = sum(x_vec)
        self.aux[M_MAT] = m_mat

        w_vec = np.zeros(self.K)
        m_mat_bar = m_mat.copy()

        if self.sbp[RHO0] > 0:
            stick_ratio = self.sbp[RHO0]
            for j in range(self.K):
                if m_mat[j,j]>0:
                    w_vec[j] = binomial(m_mat[j,j], stick_ratio/(stick_ratio+self.sbp[BETA_VEC][j]*(1-stick_ratio)))
                    m_mat_bar[j,j] = m_mat[j,j] - w_vec[j]

        self.aux[W_VEC] = w_vec
        self.aux[M_MAT_BAR] = m_mat_bar

        # last time point
        self.aux[M_MAT_BAR][0,0] += 1
        self.aux[M_MAT][0,0] += 1

    def sample_beta(self):
        try:
            prob = (self.aux[M_MAT_BAR].sum(axis=0))
            if(np.any(prob == 0)):
                print('warning')
            prob[prob == 0] += 0.01
            beta_vec = dirichlet.rvs(np.hstack((prob, self.sbp[GAMMA0])), size=1)[0]
            beta_new = beta_vec[-1]
            beta_vec = beta_vec[:-1]

            self.sbp[BETA_VEC] = beta_vec
            self.sbp[BETA_NEW] = beta_new
        except Exception as e:
            print('sample_beta failed', e)

    def update_ss(self, initialising=False):
        # NK: None,
        # N_MAT: None,
        # N_FT: None,
        nk = np.zeros(self.K)
        for k in range(self.K):
            nk[k] = np.sum(self.Z == k)

        n_mat = np.zeros((self.K, self.K))
        for i in range(self.K):
            # find indices of states that come right after state i
            indices = np.where(self.Z == i)[0] + 1  # indices of X_k

            # need to address the case for the last state in the sequence
            if self.N in indices:
                indices = np.delete(indices, np.where(indices == self.N))

            states = self.Z[indices]
            n_i = np.zeros(self.K)
            for j in range(self.K):
                n_i[j] = np.count_nonzero(states == j)
            n_mat[i,:] = n_i

        # n_ft
        n_ft = np.zeros((self.K))
        n_ft[int(self.Z[0])] += 1

        if not initialising and self.verbose:
            if np.all(self.ss[NK] == nk) != True:
                print('warning self.nk == nk) != True:')

            if np.all(self.ss[N_MAT] == n_mat) != True:
                print('warning self.nk == nk) != True:')

            if np.all(self.ss[N_FT] == n_ft) != True:
                print('warning self.nk == nk) != True:')


        self.ss[NK] = nk
        self.ss[N_MAT] = n_mat
        self.ss[N_FT] = n_ft

    def gibbs_sweep(self):
        self.sample_z()
        self.update_ss()
        self.sample_aux_vars()
        self.sample_beta()
        # sample hyper-params

    def fit(self):

        self.track = {
            ARI: np.zeros(self.iterations),
            LL: [],
            MU_TRACE: [],
            SIGMA_TRACE: [],
            A_TRACE: [],
            PIE_TRACE: [],
            TIME: []
        }

        start_time = time.time()

        for it in range(self.iterations):

            start = time.time()
            self.gibbs_sweep()

            # collect samples
            A, pie = self.sample_pi()
            mus, sigma = self.get_map_gauss_params()

            self.track[MU_TRACE].append(mus)
            self.track[SIGMA_TRACE].append(sigma)
            self.track[A_TRACE].append(A)
            self.track[PIE_TRACE].append(pie)

            if it > self.burn_in:

                if self.Z_true is not None:
                    self.track[ARI][it] = np.round(ari(self.Z_true, self.Z), 3)
                    print('iter: ', it, '   ARI: ', self.track[ARI][it], '    K: ',
                          self.K)

                if it%10 == 0:
                    _ = self.get_hmm()
                    self.track[LL].append(self.get_likelihood())
                    print('ll:  ', self.track[LL][-1])

            end = time.time()
            self.track[TIME].append(end - start)
            print('it: ', it, " -- ", end - start)

                # Calculate ARI
                #     if self.track[ARI][it] >= 0.98 and np.abs(self.track[LL][it] - self.track[LL][it-1]) < self.tol:
                #         print('ARI has reached 1 complete')
                #         break


        end_time = time.time()
        print('completed iterations in: ', end_time - start_time)

        _ = self.get_hmm()

# __________ Everything not part of gibbs sampling
    def get_map_gauss_params(self):
        """
        Return MAP estimate of the means and sigma, see murphy 4.215
        """
        sigmas = np.zeros((self.K, self.D, self.D))
        mus = np.zeros((self.K, self.D))

        k0 = self.giw[K0]
        s0 = np.copy(self.giw[S0])
        m0 = np.copy(self.giw[M0])
        nu0 = self.giw[NU0]

        Sc = s0 + (np.outer(m0, m0) * k0)

        outer_c = np.zeros((self.N, self.D, self.D))
        for i in range(self.N):
            outer_c[i, :, :] = np.outer(self.X[i], self.X[i])

        for k in range(self.K):

            kn = k0 + self.ss[NK][k]
            nun = nu0 + self.ss[NK][k]
            mc = (k0 * m0) / kn

            # calculate Sx
            Sx = np.sum(outer_c[self.Z == k], axis=0)

            # mn
            mn_top_b = np.sum(self.X[self.Z == k], axis = 0)
            mn = mc + mn_top_b / kn

            # Sn
            Sn = Sc + Sx - (np.outer(mn, mn) * kn)

            mus[k, :] = mn
            sigmas[k, :, :] = Sn / (nun + self.D + 2)

        return mus, sigmas

    def sample_pi(self):
        A = np.zeros((self.K,self.K))
        for k in range(self.K):
            prob_vec = np.hstack((self.sbp[ALPHA0]*self.sbp[BETA_VEC]+self.ss[N_MAT][k]))
            prob_vec[k] += self.sbp[KAPPA0]
            prob_vec[prob_vec<0.01] = 0.01
            A[k, :] = dirichlet.rvs(prob_vec, size=1)[0]

        prob_vec = self.sbp[ALPHA0]*self.sbp[BETA_VEC] + self.ss[N_FT]
        prob_vec[prob_vec<0.01] = 0.01
        pie = dirichlet.rvs(prob_vec, size=1)[0]
        return A, pie

    def get_hmm(self):
        # create hmm from most recent params
        if(len(self.track[A_TRACE]) > 0):
            sigma = self.track[SIGMA_TRACE][-1]
            mu = self.track[MU_TRACE][-1]
            pie = self.track[PIE_TRACE][-1]
            A = self.track[A_TRACE][-1]
        else:
            mu, sigma = self.get_map_gauss_params()
            A, pie = self.sample_pi()
        hmm_updated = GaussianHMM(self.K, covariance_type='full')
        hmm_updated.n_features, hmm_updated.covars_, hmm_updated.means_, hmm_updated.startprob_, hmm_updated.transmat_ = \
            self.D, sigma, mu, pie, A
        self.hmm = hmm_updated
        return self.hmm

    def get_likelihood(self):
        # likelihood of hmm, update hmmlearn object and using it for simplicity
        log_prob, _ = self.hmm.decode(self.X[:200])
        return log_prob

if __name__ == '__main__':
    print('demo')
    # my_hmm = InfiniteDirectSamplerHMM(loaded_data, 2, loaded_ss, iterations=40)
    # fit_vars = my_hmm.fit()



