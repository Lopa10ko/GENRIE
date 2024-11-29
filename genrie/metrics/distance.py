import torch


def compute_onnd(real: DataType, synthetic: DataType):
    """
    Calculating Outgoing Nearest Neighbour Distance to assess the diversity of the generated data
    Hiba Arnout et al. “Visual Evaluation of Generative Adversarial Networks for Time Series Data”.
    In: arXiv preprint arXiv:2001.00062 (2019).
    """
    if not isinstance(real, torch.Tensor):
        real = torch.Tensor(real)
    if not isinstance(synthetic, torch.Tensor):
        synthetic = torch.Tensor(synthetic)

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


def compute_innd(real: DataType, synthetic: DataType):
    """
    Calculating Incoming Nearest Neighbour Distance to assess the authenticity of the generated data
    Hiba Arnout et al. “Visual Evaluation of Generative Adversarial Networks for Time Series Data”.
    In: arXiv preprint arXiv:2001.00062 (2019).
    """
    if not isinstance(real, torch.Tensor):
        real = torch.Tensor(real)
    if not isinstance(synthetic, torch.Tensor):
        synthetic = torch.Tensor(synthetic)

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
