import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from genrie.generation.cgan import Generator, Discriminator
from genrie.utils.loader import UCRLoader
from genrie.data.data_adapter import DataStore
from genrie.metrics.metric import DistanceMetric

if __name__ == '__main__':

    loader = UCRLoader('EMOPain')
    x_train, y_train, x_test, y_test = loader.load()
    data = DataStore(features=x_test, target=y_test)
    print(data.features.shape, data.target.shape, data.covmats.shape)
    ortho_covmats = data.get_synthetic_covmats(mode='ortho')
    noise_covmats = data.get_synthetic_covmats(mode='noise')
    n_samples, n_channels, ts_length = data.features.shape
    latent_dim = ts_length
    n_epochs = 100
    batch_size = 32
    learning_rate = 0.0002
    X_tensor = torch.tensor(data.features, dtype=torch.float32)
    covmats_tensor = torch.tensor(data.covmats, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, covmats_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator(latent_dim=latent_dim, ts_length=ts_length, n_channels=n_channels)
    discriminator = Discriminator(n_channels=n_channels, ts_length=ts_length)
    criterion = nn.BCELoss()

    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    pbar = tqdm(range(n_epochs))

    for epoch in pbar:
        for i, (real_ts, real_cov) in enumerate(dataloader):
            d_optimizer.zero_grad()
            real_labels = torch.ones(real_ts.size(0), 1)
            fake_labels = torch.zeros(real_ts.size(0), 1)

            d_real = discriminator(real_ts, real_cov)
            d_real_loss = criterion(d_real, real_labels)

            noise = torch.randn(real_ts.size(0), latent_dim)
            fake_ts = generator(noise, real_cov)
            d_fake = discriminator(fake_ts.detach(), real_cov)
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            output = discriminator(fake_ts, real_cov)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            g_optimizer.step()

            pbar.write(f'D-Loss: {d_loss.item():.4f}, G-Loss: {g_loss.item():.4f}')
    pbar.close()

    ortho_covmats = data.get_synthetic_covmats(mode='ortho')

    ortho_samples = np.zeros(data.features.shape)

    for i, ortho_matrix in enumerate(ortho_covmats):
        with torch.no_grad():
            new_covmat = torch.tensor(ortho_matrix, dtype=torch.float32)
            z = torch.randn(1, latent_dim)
            generated_sample = generator(z, new_covmat.unsqueeze(0)).detach().numpy()
            ortho_samples[i] = generated_sample

    print(DistanceMetric()(data.features, ortho_samples))
