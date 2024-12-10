from collections.abc import Mapping
from typing import Sequence, Optional, Callable

import numpy as np

from genrie.data.data_adapter import DataType
from genrie.metrics.distance import compute_onnd, compute_innd


AVAILABLE_METRICS = {
    'onnd': compute_onnd,
    'innd': compute_innd,
}


class DistanceMetric:
    def __init__(self, metrics: Optional[Mapping[str, Callable]] = None):
        self.metrics = metrics if metrics is not None else AVAILABLE_METRICS

    def __call__(self, real: DataType, synthetic: DataType):
        metric_dict = {}
        for metric_name, metric in self.metrics.items():
            try:
                metric_value = metric(real, synthetic)
            except Exception as _:
                metric_value = np.nan
            metric_dict[metric_name] = metric_value

        return metric_dict
