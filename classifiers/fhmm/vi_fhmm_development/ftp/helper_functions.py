import numpy as np

tiny = np.exp(-700)

def number_to_base(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def get_states_assignments(no, no_states, no_chains):
    states = number_to_base(no, no_states)
    if len(states) < no_chains:
        states = np.append(np.zeros((no_chains - len(states)), dtype=int), states)
    return states


def calc_mf_like(X, T, mf, M, K, Mu, Cov, P, Pi):
    p = X.shape[1]
    kpm = np.power(K, M)

    k1 = np.power((2 * np.pi), (-p / 2))

    dd = np.zeros((kpm, M), dtype=int)
    Mf = np.ones((T, kpm))
    Mub = np.zeros((kpm, p))

    for i in range(kpm):
        dd[i, :] = get_states_assignments(i, K, M)
        for j in range(M):
            Mub[i, :] = Mub[i, :] + Mu[j * K + dd[i, j], :]
            Mf[:, i] = Mf[:, i] * mf[:, j * K + dd[i, j]]

    logPi = np.log(np.where(Pi < tiny, tiny, Pi))
    logP = np.log(np.where(P < tiny, tiny, P))
    logmf = np.log(np.where(mf < tiny, tiny, mf))

    lik = 0

    iCov = np.linalg.inv(Cov)
    k2 = k1 / np.sqrt(np.linalg.det(Cov))
    for i in range(kpm):
        d = (np.ones((T, 1)) @ Mub[i, :])[..., np.newaxis] - X
        lik = lik - 0.5 * np.sum(Mf[:, i] * np.sum(((d @ iCov) * d), axis=1))  # will break for

    lik = lik + T * np.log(k2)

    lik + (mf[0, :] @ np.resize(logPi.T, logPi.shape[0] + logPi.shape[1])) - np.sum(mf[0, :] * logmf[0, :])

    for i in range(1, T):
        d1 = i
        d0 = i - 1
        for j in range(M):
            d2 = np.arange(j * K, j * K + K)
            lik = lik + np.sum(mf[d0, [d2]] * (mf[d1, d2] @ logP[d2, :].T)) - np.sum(mf[d1, d2] * logmf[d1, d2])

    return lik