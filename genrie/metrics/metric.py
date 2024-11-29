from genrie.data.data_adapter import DataType
from genrie.metrics.distance import compute_onnd, compute_innd

class DistanceMetric:
    metrics = {
        'onnd': compute_onnd,
        'innd': compute_innd,
    }

class Metric:
    def __call__(self, real: DataType, synthetic: DataType):
        self.metrics_dict = {
            metric_name: metric_func(real, synthetic)
            for metric_name, metric_func in self.metrics.items()
        }

