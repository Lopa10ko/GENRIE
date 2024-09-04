import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance

if __name__ == '__main__':
    rs = np.random.RandomState(42)
    n_matrices, n_channels, n_times = 10, 5, 2000
    var = 2.0 + 0.1 * rs.randn(n_matrices, n_channels)
    A = 2 * rs.rand(n_channels, n_channels) - 1
    A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
    true_covs = np.empty(shape=(n_matrices, n_channels, n_channels))
    X = np.empty(shape=(n_matrices, n_channels, n_times))
    for i in range(n_matrices):
        true_covs[i] = A @ np.diag(var[i]) @ A.T
        X[i] = rs.multivariate_normal(np.array([0.0] * n_channels), true_covs[i], size=n_times).T

    estimators = ['cov', 'lwf', 'oas', 'sch', 'scm']
    w_len = np.linspace(10, n_times, 20, dtype=int)
    dfd = list()
    for est in estimators:
        for wl in w_len:
            est_covs = Covariances(estimator=est).transform(X[:, :, :wl])
            dists = distance(est_covs, true_covs, metric="riemann")
            dfd.extend([dict(estimator=est, wlen=wl, dist=d) for d in dists])
    dfd = pd.DataFrame(dfd)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set(xscale="log")
    sns.lineplot(data=dfd, x="wlen", y="dist", hue="estimator", ax=ax)
    ax.set_title("Distance to groundtruth covariance matrix")
    ax.set_xlabel("Number of time samples")
    ax.set_ylabel(r"$\delta(\Sigma, \hat{\Sigma})$")
    plt.tight_layout()
    plt.show()
