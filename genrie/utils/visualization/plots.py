from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from pyriemann.embedding import SpectralEmbedding
from pyriemann.estimation import Covariances, Shrinkage
from sklearn.decomposition import PCA
from torch import tensor


def plot_sns_heatmap(matrix: np.ndarray):
    with plt.ion():
        fig, axes = plt.subplots(1, 1)
        sns.heatmap(matrix, cmap="crest", ax=axes)
        plt.tight_layout()
        plt.show()


def plot_covariance_surface(covmat: np.ndarray, synthetic_covmat: np.ndarray):
    x = np.arange(covmat.shape[1])
    x_axes, y_axes = np.meshgrid(x, x)
    surf = go.Surface(x=x_axes, y=y_axes, z=covmat,
                      colorscale='Blues', name='original covmat')
    surf_manipulated = go.Surface(x=x_axes, y=y_axes, z=synthetic_covmat,
                                  colorscale='Reds', opacity=0.8, name='synthetic covmat')
    layout = go.Layout(
        title='Covariance Function Surface Plot',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Covariance'
        ),
        width=800,
        height=800
    )
    fig = go.Figure(data=[surf, surf_manipulated], layout=layout)
    fig.show()


def plot_spectral_embeddings_comparison(features: np.ndarray,
                                        new_features: np.ndarray,
                                        spd_space: Covariances,
                                        shrinkage: Shrinkage) -> None:
    lapl = SpectralEmbedding(metric='riemann', n_components=3)
    fig = go.Figure()
    for label, features_arr in zip(['original data', 'synthetical data'], [features, new_features]):
        covmats = spd_space.transform(features_arr)
        covmats = shrinkage.transform(covmats)
        reduced_data = lapl.fit_transform(X=covmats)
        df = pd.DataFrame(reduced_data, columns=['EMB1', 'EMB2', 'EMB3'])
        fig.add_trace(go.Scatter3d(x=df['EMB1'], y=df['EMB2'], z=df['EMB3'],
                                   mode='markers', marker=dict(size=5), name=label))
        fig.update_layout(scene=dict(xaxis_title='EMB1',
                                     yaxis_title='EMB2',
                                     zaxis_title='EMB3'),
                          title='Comparison of spectral embeddings of old and transformed features')
    fig.show()


def plot_pca_comparison(features: np.ndarray, new_features: np.ndarray) -> None:
    pca = PCA(n_components=3)
    fig = go.Figure()
    for label, features_arr in zip(['original data', 'synthetical data'], [features, new_features]):
        features_tensor = tensor(features_arr)
        flattened_data = features_tensor.reshape(features_tensor.shape[0], -1)
        reduced_data = pca.fit_transform(flattened_data)
        df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2', 'PC3'])
        fig.add_trace(go.Scatter3d(x=df['PC1'], y=df['PC2'], z=df['PC3'],
                                   mode='markers', marker=dict(size=5), name=label))
        fig.update_layout(scene=dict(xaxis_title='PC1',
                                     yaxis_title='PC2',
                                     zaxis_title='PC3'),
                          title='Comparison of pca decompositions of old and transformed features')
    fig.show()


def plot_multivariate_data_comparison(features: np.ndarray,
                                      new_features: np.ndarray,
                                      sample_id: int = 0,
                                      channel_ids: Optional[Sequence[int]] = None,
                                      show_multiplot: bool = True) -> None:
    with plt.ion():
        n_samples, n_channels, ts_length = features.shape
        sample, new_sample = features[sample_id], new_features[sample_id]
        channel_ids = range(n_channels) if channel_ids is None else channel_ids
        fig = plt.figure(figsize=(15, 10))
        plt.title(f'Sample {sample_id}')

        for i, channel_id in enumerate(channel_ids):
            if show_multiplot:
                plt.subplot(len(channel_ids), 1, i + 1)
            for cmap, label, data in zip([plt.get_cmap('Blues'), plt.get_cmap('Reds')],
                                         ['orig', 'synth'],
                                         [sample, new_sample]):
                plt.plot(data[channel_id],
                         color=cmap(np.random.randint(50, 200)),
                         label=f'{label} - {channel_id}')

        plt.legend()
        plt.tight_layout()
        plt.show()
