import numpy as np

from genrie.data.data_adapter import DataType
from genrie.metrics.distance import compute_onnd, compute_innd


AVAILABLE_METRICS = {
    'onnd': compute_onnd,
    'innd': compute_innd,
}


class DistanceMetric:
    @staticmethod
    def __call__(real: DataType, synthetic: DataType):
        metric_dict = {}
        for metric_name, metric in AVAILABLE_METRICS.items():
            try:
                metric_value = metric(real, synthetic)
            except Exception as _:
                metric_value = np.nan
            metric_dict[metric_name] = metric_value

        return metric_dict
