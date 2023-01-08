import numpy as np


def pmhmm(X, M=2, K=4, cyc=100, tol=0.0001, iters=10):

    p = X.shape[1]
    N = X.shape[0]
    T = N

    Cov = np.diag(np.diag(np.cov(X)))
    XX = np.dot(X.T, X) / (np.dot(N, T))

    # X reordered with TN*p rather than NT*p :
    Xalt = []
    ai = arange(1, dot(N, T))
    for i in arange(1, T).reshape(-1):
        indxi = rem(ai - i, T) == 0
        Xalt = concat([[Xalt], [X(indxi, arange())]])

    Mu = dot(randn(dot(M, K), p), sqrtm(Cov)) / M + dot(ones(dot(K, M), 1), mean(X)) / M
    Pi = rand(K, M)
    Pi = cdiv(Pi, csum(Pi))
    P = rand(dot(K, M), K)
    P = rdiv(P, rsum(P))
    LL = []
    lik = 0
    TL = []
    gamma = zeros(dot(N, T), dot(K, M))

    k1 = (dot(2, pi)) ** (- p / 2)
    mf = ones(dot(N, T), dot(M, K)) / K
    h = ones(dot(N, T), dot(M, K)) / K
    alpha = zeros(dot(N, T), dot(M, K))
    beta = zeros(dot(N, T), dot(M, K))
    logmf = log(mf)
    exph = exp(h)
    for cycle in arange(1, cyc).reshape(-1):
        #### FORWARD-BACKWARD ### MF E STEP
        Gamma = []
        GammaX = zeros(dot(M, K), p)
        Eta = zeros(dot(K, M), dot(K, M))
        Xi = zeros(dot(M, K), K)
        iCov = inv(Cov)
        k2 = k1 / sqrt(det(Cov))
        itermf = copy(iters)
        for l in arange(1, iters).reshape(-1):
            mfo = copy(mf)
            logmfo = copy(logmf)
            for i in arange(1, T).reshape(-1):
                d2 = arange(dot((i - 1), N) + 1, dot(i, N))
                d = Xalt(d2, arange())
                # compute yhat (N*p);
                yhat = dot(mf(d2, arange()), Mu)
                for j in arange(1, M).reshape(-1):
                    d1 = arange(dot((j - 1), K) + 1, dot(j, K))
                    Muj = Mu(d1, arange())
                    h[d2, d1] = (
                                dot(dot(Muj, iCov), (d - yhat).T) + dot(dot(dot(Muj, iCov), Muj.T), mf(d2, d1).T) - dot(
                            dot(0.5, diag(dot(dot(Muj, iCov), Muj.T))), ones(1, N))).T
                    h[d2, d1] = h(d2, d1) - dot(max(h(d2, d1).T).T, ones(1, K))
            exph = exp(h)
            scale = zeros(dot(T, N), M)
            for j in arange(1, M).reshape(-1):
                d1 = arange(dot((j - 1), K) + 1, dot(j, K))
                d2 = arange(1, N)
                alpha[d2, d1] = multiply(exph(d2, d1), (dot(ones(N, 1), Pi(arange(), j).T)))
                scale[d2, j] = rsum(alpha(d2, d1)) + tiny
                alpha[d2, d1] = rdiv(alpha(d2, d1), scale(d2, j))
                for i in arange(2, T).reshape(-1):
                    d2 = arange(dot((i - 1), N) + 1, dot(i, N))
                    alpha[d2, d1] = multiply((dot(alpha(d2 - N, d1), P(d1, arange()))), exph(d2, d1))
                    scale[d2, j] = rsum(alpha(d2, d1)) + tiny
                    alpha[d2, d1] = rdiv(alpha(d2, d1), scale(d2, j))
                d2 = arange(dot((T - 1), N) + 1, dot(T, N))
                beta[d2, d1] = rdiv(ones(N, K), scale(d2, j))
                for i in arange(T - 1, 1, - 1).reshape(-1):
                    d2 = arange(dot((i - 1), N) + 1, dot(i, N))
                    beta[d2, d1] = dot((multiply(beta(d2 + N, d1), exph(d2 + N, d1))), (P(d1, arange()).T))
                    beta[d2, d1] = rdiv(beta(d2, d1), scale(d2, j))
                mf[arange(), d1] = (multiply(alpha(arange(), d1), beta(arange(), d1)))
                mf[arange(), d1] = rdiv(mf(arange(), d1), rsum(mf(arange(), d1)) + tiny)
            logmf = log(mf + multiply((mf == 0), tiny))
            delmf = sum(sum(multiply(mf, logmf))) - sum(sum(multiply(mf, logmfo)))
            if (delmf < dot(dot(N, T), 1e-06)):
                itermf = copy(l)
                break
        # calculating mean field log likelihood
        if (nargout >= 5):
            oldlik = copy(lik)
            lik = calcmflike(Xalt, T, mf, M, K, Mu, Cov, P, Pi)
        gamma = copy(mf)
        Gamma = copy(gamma)
        Eta = dot(gamma.T, gamma)
        gammasum = sum(gamma)
        for j in arange(1, M).reshape(-1):
            d2 = arange(dot((j - 1), K) + 1, dot(j, K))
            Eta[d2, d2] = diag(gammasum(d2))
        GammaX = dot(gamma.T, Xalt)
        Eta = (Eta + Eta.T) / 2
        Xi = zeros(dot(M, K), K)
        for i in arange(1, T - 1).reshape(-1):
            d1 = arange(dot((i - 1), N) + 1, dot(i, N))
            d2 = d1 + N
            for j in arange(1, M).reshape(-1):
                jK = arange(dot((j - 1), K) + 1, dot(j, K))
                t = multiply(P(jK, arange()), (dot(alpha(d1, jK).T, (multiply(beta(d2, jK), exph(d2, jK))))))
                Xi[jK, arange()] = Xi(jK, arange()) + t / sum(ravel(t))
        LL = concat([LL, lik])
        if (nargout >= 6):
            truelik = calclike(X, T, M, K, Mu, Cov, P, Pi)
            TL = concat([TL, truelik])
            fprintf('cycle %i mf iters %i log like= %f true log like= %f', cycle, itermf, lik, truelik)
        else:
            if (nargout == 5):
                fprintf('cycle %i mf iters %i log likelihood = %f ', cycle, itermf, lik)
            else:
                fprintf('cycle %i mf iters %i ', cycle, itermf)
        if (nargout >= 5):
            if (cycle <= 2):
                likbase = copy(lik)
            else:
                if (lik < oldlik - 2):
                    fprintf('v')
                else:
                    if (lik < oldlik):
                        fprintf('v')
                    else:
                        if ((lik - likbase) < dot((1 + tol), (oldlik - likbase))):
                            fprintf('\\n')
                            break
        fprintf('\\n')
        # outputs -- using SVD as generally ill-conditioned (Mu=pinv(Eta)*GammaX);
        U, S, V = svd(Eta, nargout=3)
        Si = zeros(dot(K, M), dot(K, M))
        for i in arange(1, dot(K, M)).reshape(-1):
            if (S(i, i) < dot(dot(max(size(S)), norm(S)), 0.001)):
                Si[i, i] = 0
            else:
                Si[i, i] = 1 / S(i, i)
        Mu = dot(dot(dot(V, Si), U.T), GammaX)
        Cov = XX - dot(dot(GammaX.T, pinv(Eta)), GammaX) / (dot(N, T))
        Cov = (Cov + Cov.T) / 2
        dCov = det(Cov)
        while (dCov < 0):
            fprintf('\\nAbort: covariance problem \\n')
            return Mu, Cov, P, Pi, LL, TL

        for i in arange(1, dot(K, M)).reshape(-1):
            d1 = sum(Xi(i, arange()))
            if (d1 == 0):
                P[i, arange()] = ones(1, K) / K
            else:
                P[i, arange()] = Xi(i, arange()) / d1
        Pi = reshape(csum(Gamma(arange(1, N), arange())), K, M) / N
