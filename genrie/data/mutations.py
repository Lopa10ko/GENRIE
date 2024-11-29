import numpy as np

from abc import ABC, abstractmethod
from typing import Callable
from scipy.stats import ortho_group

from genrie.data.data_adapter import DataType


class SingleMutation(Callable, ABC):
    @abstractmethod
    def __call__(self, mats: DataType, **kwargs) -> DataType:
        pass


class OrthoMutation(SingleMutation):
    def __call__(self, mats: DataType, **kwargs) -> DataType:
        ortho_matrix = ortho_group.rvs(dim=mats.shape[0])
        new_mats = np.array([ortho_matrix @ m for m in mats])
        return new_mats


class NoiseMutation(SingleMutation):
    def __call__(self, mats: DataType, **kwargs) -> DataType:
        mean = kwargs.get('mean', 0)
        stdev = kwargs.get('stdev', 1e-1)
        new_mats = np.array([np.random.normal(mean, stdev, size=m.shape) + m for m in mats])
        return new_mats


def get_mutation_by_mode(mode: str):
    mutations = {
        'ortho': OrthoMutation(),
        'noise': NoiseMutation(),
    }
    if mode not in mutations.keys():
        raise NotImplementedError(f'Mutation mode {mode} not implemented')
    return mutations[mode]
