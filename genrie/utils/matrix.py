import numpy as np
import torch
from scipy.linalg import svd, norm, eigvals, cholesky, LinAlgError


def get_nearest_positive_defined(matrix):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (matrix + matrix.T) / 2
    _, s, V = svd(B)
    A2 = (B + np.dot(V.T, np.dot(np.diag(s), V))) / 2
    A3 = (A2 + A2.T) / 2
    if is_poisitive_defined(A3):
        return A3
    spacing = np.spacing(norm(matrix))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(matrix.shape[0])
    k = 1
    while not is_poisitive_defined(A3):
        mineig = np.min(np.real(eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3


def is_poisitive_defined(B):
    try:
        _ = cholesky(B)
        return True
    except LinAlgError:
        return False


def gaussian_kernel(x, y, sigmas):
    """
    Compute the Gaussian kernel between two tensors.
    """
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = torch.cdist(x, y) ** 2
    s = torch.matmul(beta, dist.view(1, -1))
    return torch.exp(-s)


def data_to_tensor(func):
    """
    Decorator to convert input data to PyTorch tensors.
    """
    def wrapper(*args, **kwargs):
        tensor_args = []
        for value in args:
            if isinstance(value, (list, tuple, np.ndarray)):
                tensor_args.append(torch.tensor(value, dtype=torch.float32))
            elif isinstance(value, torch.Tensor):
                tensor_args.append(value)
            else:
                raise ValueError(f"Data type: {type(value)} is not supported")
        return func(*tensor_args, **kwargs)
    return wrapper
