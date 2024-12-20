import torch
from scipy.stats import wasserstein_distance

from genrie.data.data_adapter import DataType
from genrie.utils.matrix import data_to_tensor, gaussian_kernel


@data_to_tensor
def compute_onnd(real: DataType, synthetic: DataType):
    """
    Calculating Outgoing Nearest Neighbour Distance to assess the diversity of the generated data
    Hiba Arnout et al. “Visual Evaluation of Generative Adversarial Networks for Time Series Data”.
    In: arXiv preprint arXiv:2001.00062 (2019).
    """
    n_samples, n_channels, ts_length = real.shape
    n_samples_synth, n_channels_synth, ts_length_synth = synthetic.shape
    assert n_channels == n_channels_synth
    assert ts_length == ts_length_synth

    # minimum MSE for each real sample
    min_mse = torch.zeros(n_samples, device=real.device)

    for i in range(n_samples):
        samplewise_diff = synthetic - real[i].unsqueeze(0)
        samplewise_mse = torch.norm(samplewise_diff, dim=2).mean(dim=1)
        min_mse[i] = torch.min(samplewise_mse)

    metric = min_mse.mean()
    return metric.cpu().numpy()


@data_to_tensor
def compute_innd(real: DataType, synthetic: DataType):
    """
    Calculating Incoming Nearest Neighbour Distance to assess the authenticity of the generated data
    Hiba Arnout et al. “Visual Evaluation of Generative Adversarial Networks for Time Series Data”.
    In: arXiv preprint arXiv:2001.00062 (2019).
    """
    n_samples, n_channels, ts_length = real.shape
    n_samples_synth, n_channels_synth, ts_length_synth = synthetic.shape
    assert n_channels == n_channels_synth
    assert ts_length == ts_length_synth

    # minimum MSE for each synthetic sample
    min_mse = torch.zeros(n_samples_synth, device=real.device)

    for i in range(n_samples_synth):
        samplewise_diff = real - synthetic[i].unsqueeze(0)
        samplewise_mse = torch.norm(samplewise_diff, dim=2).mean(dim=1)
        min_mse[i] = torch.min(samplewise_mse)

    metric = min_mse.mean()
    return metric.cpu().numpy()


@data_to_tensor
def compute_mmd(real: DataType, synthetic: DataType):
    """
    Calculating Maximum Mean Discrepancy.
    """
    n_samples, n_channels, ts_length = real.shape
    n_samples_synth, n_channels_synth, ts_length_synth = synthetic.shape
    assert n_samples == n_samples_synth
    assert n_samples != 1
    assert n_channels == n_channels_synth
    assert ts_length == ts_length_synth

    sigmas = torch.tensor([2.0, 5.0, 10.0, 20.0, 40.0, 80.0], dtype=torch.float32)

    real_k = gaussian_kernel(real, real, sigmas).sum()
    pair_k = gaussian_kernel(real, synthetic, sigmas).sum()
    synthetic_k = gaussian_kernel(synthetic, synthetic, sigmas).sum()

    metric = (real_k + synthetic_k) / (n_samples * (n_samples - 1))
    metric -= 2 * pair_k / n_samples ** 2

    return metric.cpu().numpy()


@data_to_tensor
def compute_wasserstein(real: DataType, synthetic: DataType):
    """
    Calculating mean Wasserstein Distance between two distributions
    induced by real and synthetic time series.
    """
    n_samples, n_channels, ts_length = real.shape
    n_samples_synth, n_channels_synth, ts_length_synth = synthetic.shape
    assert n_channels == n_channels_synth
    assert ts_length == ts_length_synth

    min_n_samples = min(n_samples, n_samples_synth)
    distance = torch.zeros((min_n_samples, n_channels))

    for id_sample in range(min_n_samples):
        for id_channel in range(n_channels):
            distance[id_sample][id_channel] = wasserstein_distance(real[id_sample][id_channel],
                                                                   synthetic[id_sample][id_channel])

    metric = torch.mean(distance)
    return metric.cpu().numpy()
