import os
import torch
import zipfile
import urllib.request
import numpy as np
import pandas as pd

from pathlib import Path

from torch import nn, optim
from tqdm import tqdm
from sktime.datasets import load_from_tsfile_to_dataframe

from genrie.data.data_adapter import DataStore
from genrie.generation.gan import Generator, Discriminator
from genrie.metrics.metric import DistanceMetric
from genrie.utils.path_lib import EXAMPLES_PATH


def get_eigen_worms_dataset():
    EXAMPLES_DATA_PATH = Path(EXAMPLES_PATH, 'data')
    os.makedirs(EXAMPLES_DATA_PATH, exist_ok=True)
    DATA_URL = f'http://www.timeseriesclassification.com/aeon-toolkit/EigenWorms.zip'
    ZIP_PATH = Path(EXAMPLES_DATA_PATH, 'EigenWorms.zip')
    DATA_PATH = Path(EXAMPLES_DATA_PATH, 'EigenWorms')

    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        zipfile.ZipFile(ZIP_PATH).extractall(DATA_PATH)

    x_train, y_train = load_from_tsfile_to_dataframe(str(Path(DATA_PATH, 'EigenWorms_TRAIN.ts')),
                                                     return_separate_X_and_y=True)
    x_test, y_test = load_from_tsfile_to_dataframe(str(Path(DATA_PATH, 'EigenWorms_TEST.ts')),
                                                   return_separate_X_and_y=True)

    shuffled_idx = np.arange(x_train.shape[0])
    np.random.shuffle(shuffled_idx)
    if isinstance(x_train, pd.DataFrame):
        x_train = x_train.iloc[shuffled_idx, :]
    else:
        x_train = x_train[shuffled_idx, :]
    y_train = y_train[shuffled_idx]

    if isinstance(x_train.iloc[0, 0], pd.Series):
        def convert(arr):
            return np.array([d.values for d in arr])
        x_train = np.apply_along_axis(convert, 1, x_train)
        x_test = np.apply_along_axis(convert, 1, x_test)

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    train_data, test_data = get_eigen_worms_dataset()

    data = DataStore(features=test_data[0], target=test_data[1])
    ortho_covmats = data.get_synthetic_covmats(mode='ortho')

    n_samples, n_channels, ts_length = data.features.shape
    latent_dim = 4 * ts_length
    n_epochs = 20
    batch_size = 6
    learning_rate = 0.0002

    X_tensor = torch.tensor(data.features, dtype=torch.float32)
    covmats_tensor = torch.tensor(data.covmats, dtype=torch.float32)

    generator = Generator(latent_dim, n_channels, ts_length)
    discriminator = Discriminator(n_channels, ts_length)
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    pbar = tqdm(range(n_epochs))

    for epoch in pbar:
        for i in range(0, n_samples, batch_size):
            real_samples = X_tensor[i:i + batch_size]
            real_covmats = covmats_tensor[i:i + batch_size]

            valid = torch.ones(real_samples.size(0), 1)
            fake = torch.zeros(real_samples.size(0), 1)

            z = torch.randn(real_samples.size(0), latent_dim)  # Random noise
            gen_samples = generator(z, real_covmats)  # Generate fake samples

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_samples, real_covmats), valid)
            fake_loss = adversarial_loss(discriminator(gen_samples.detach(), real_covmats), fake)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            g_loss = adversarial_loss(discriminator(gen_samples, real_covmats), valid)
            g_loss.backward()
            optimizer_G.step()

        pbar.write(f'D-Loss: {d_loss.item():.4f}, G-Loss: {g_loss.item():.4f}')
    pbar.close()

    ortho_samples = np.zeros(data.features.shape)

    for i, ortho_matrix in enumerate(ortho_covmats):
        with torch.no_grad():
            new_covmat = torch.tensor(ortho_matrix, dtype=torch.float32)
            z = torch.randn(1, latent_dim)
            generated_sample = generator(z, new_covmat.unsqueeze(0)).detach().numpy()
            ortho_samples[i] = generated_sample

    print(DistanceMetric()(data.features, ortho_samples))
