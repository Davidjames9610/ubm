from numba import jit
from scipy.special import gammaln
import numpy as np
import math
import jax.numpy as jnp


# mv students t
# @jit(nopython=True)
def multivariate_students_t_numba(x, mt, kt, nut, St, D):

    mu = mt
    shape = St * (kt + 1.)/(kt*(nut - D + 1.))
    v = int(nut - D + 1.)
    delta = x - mu

    logdet_covar = jnp.linalg.slogdet(shape)[1]
    inv_covar = jnp.linalg.pinv(shape)

    return_value = gammaln((v + D) / 2) - gammaln(v / 2) - D / 2 * jnp.log(v) - D / 2 * jnp.log(
        jnp.pi) - logdet_covar / 2 - (v + D) / 2 * jnp.log(1 + 1. / v * jnp.dot(jnp.dot(delta, inv_covar), delta))

    return return_value

def multivariate_students_t(x, mu, shape, v, D):
    delta = x - mu
    v = int(v)

    logdet_covar = np.linalg.slogdet(shape)[1]
    inv_covar = np.linalg.pinv(shape)
    try:
        return_value = gammaln((v + D) / 2) - gammaln((v) / 2) - D/ 2. * np.log(v) - D / 2. * np.log(
            np.pi) - logdet_covar / 2 - \
                       (v + D) / 2. * math.log(1 + 1./v * np.dot(np.dot(delta, inv_covar), delta))
        return return_value

    except Exception as e:
        print('_multivariate_students_t', e)
        # shape = ensure_symmetric_positive_semidefinite(shape)
        #
        # delta = x - mu
        # v = int(v)
        #
        # logdet_covar = np.linalg.slogdet(shape)[1]
        # inv_covar = np.linalg.inv(shape)
        #
        # return_value = gammaln((v + D) / 2) - gammaln((v) / 2) - D / 2. * np.log(v) - D / 2. * np.log(
        #     np.pi) - logdet_covar / 2 - \
        #                (v + D) / 2. * math.log(1 + 1. / v * np.dot(np.dot(delta, inv_covar), delta))
        # return return_value

def student_t_giw(x, mt, kt, nut, St, D):

    loc = mt
    shape = St * (kt + 1.)/(kt*(nut - D + 1.))
    df = (nut - D + 1.)
    return multivariate_students_t(x, loc, shape, df, D)

def is_symmetric_positive_semidefinite(matrix):
    # Check for symmetry
    if not np.all(matrix == matrix.T):
        return False

    # Check for positive semidefiniteness
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def ensure_symmetric_positive_semidefinite(A):
    # Ensure symmetry
    A = (A + A.T) / 2.0

    # Ensure positive semidefiniteness
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < 0] = 0
    A = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return A