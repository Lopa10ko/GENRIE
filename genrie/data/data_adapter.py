import torch
import numpy as np

from itertools import product
from logging import Logger
from typing import Union
from dataclasses import dataclass, field
from pyriemann.estimation import Covariances, Shrinkage
from pyriemann.utils.distance import distance
from scipy.linalg import logm, expm

from genrie.data.mutations import SingleMutation, get_mutation_by_mode

DataType = Union[np.ndarray, torch.Tensor]
logger = Logger(name='genrie.data')



@dataclass
class DataStore:
    features: DataType = field(init=True)
    target: DataType = field(init=True)
    spd_space: Covariances = Covariances(estimator='scm')
    shrinkage: Shrinkage = Shrinkage()
    n_samples: int = field(init=False, default=0)
    n_channels: int = field(init=False, default=0)
    ts_length: int = field(init=False, default=0)
    covmats: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.n_samples, self.n_channels, self.ts_length = self.features.shape
        covmats = self.spd_space.fit_transform(self.features, self.target)
        covmats = self.shrinkage.fit_transform(covmats)
        self.covmats = covmats

    def get_synthetic_covmats(self, **kwargs) -> np.ndarray:
        mode = kwargs.get('mode', 'ortho')
        mutation = get_mutation_by_mode(mode)
        new_covmats, diff = self.__mutate_mats(mutation, self.covmats)
        logger.debug(msg=f'Mutated covmats using {mode} mode with {diff.sum()} distance diff')
        return new_covmats

    @staticmethod
    def __mutate_mats(mutation: SingleMutation, mats: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        tangent_mats = np.array(list(map(logm, mats)))

        distances = np.array(
            [distance(a, b, metric='euclid') for a, b in product(tangent_mats, tangent_mats)]
        ).reshape(tangent_mats.shape[0], -1)
        new_mats = mutation(tangent_mats, **kwargs)
        new_distances = np.array(
            [distance(a, b, metric='euclid') for a, b in product(new_mats, new_mats)]
        ).reshape(new_mats.shape[0], -1)

        new_mats = np.array(list(map(expm, new_mats)))

        return new_mats, new_distances - distances

