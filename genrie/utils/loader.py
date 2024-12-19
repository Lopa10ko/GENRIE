import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional

from genrie.utils.path_lib import EXAMPLES_PATH
from sktime.datasets import load_from_tsfile_to_dataframe


@dataclass
class UCRLoader:
    dataset_name: str
    data_path: Union[str, Path] = field(default=EXAMPLES_PATH)

    def load(self,
             cwd: Optional[Union[str, Path]] = None,
             request_url: Optional[str] = None,
             with_preprocessing: bool = True):
        if cwd is None:
            cwd = Path(EXAMPLES_PATH, 'data')
        os.makedirs(cwd, exist_ok=True)

        if request_url is None:
            request_url = f'http://www.timeseriesclassification.com/aeon-toolkit/{self.dataset_name}.zip'

        zip_path = Path(cwd, f'{self.dataset_name}.zip')
        data_path = Path(cwd, f'{self.dataset_name}')

        if not os.path.exists(data_path):
            urllib.request.urlretrieve(request_url, zip_path)
            with zipfile.ZipFile(zip_path) as file:
                file.extractall(data_path)
            os.remove(zip_path)

        if not list(filter(lambda file_name: file_name.endswith('ts'), os.listdir(data_path))):
            raise ValueError(f'There is no .ts files to read in {data_path}')

        x_train, y_train, x_test, y_test = self.load_from_tsfile(data_path)

        if with_preprocessing:
            return self.preprocess_inputs(x_train, y_train, x_test, y_test)
        return x_train, y_train, x_test, y_test


    def load_from_tsfile(self, data_path: Path):
        x_train, y_train = load_from_tsfile_to_dataframe(
            str(Path(data_path, f'{self.dataset_name}_TRAIN.ts')),
            return_separate_X_and_y=True)
        x_test, y_test = load_from_tsfile_to_dataframe(
            str(Path(data_path, f'{self.dataset_name}_TEST.ts')),
            return_separate_X_and_y=True)
        return x_train, y_train, x_test, y_test


    @staticmethod
    def preprocess_inputs(x_train, y_train, x_test, y_test):
        shuffled_idx = np.arange(x_train.shape[0])
        np.random.shuffle(shuffled_idx)
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.iloc[shuffled_idx, :]
        else:
            x_train = x_train[shuffled_idx, :]
        y_train = y_train[shuffled_idx]

        if isinstance(x_train.iloc[0, 0], pd.Series):
            def convert(arr):
                return np.array([d.values for d in arr], dtype=float)

            x_train = np.apply_along_axis(convert, 1, x_train)
            x_test = np.apply_along_axis(convert, 1, x_test)
        return x_train, y_train, x_test, y_test
