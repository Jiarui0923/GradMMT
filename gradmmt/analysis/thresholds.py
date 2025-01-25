import numpy as np

class Thresholds:
    @staticmethod
    def percentile(ratio=0.15):
        return lambda weights : np.sort(weights.flatten())[-int(np.ceil((weights.shape[0] if len(weights.shape) == 1 else weights.shape[1]) * ratio))]
        # return lambda weights : (np.max(weights) - np.min(weights)) * ratio + np.min(weights)