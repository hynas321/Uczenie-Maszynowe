from typing import Optional

import numpy as np

class MinMaxScaler:
    def __init__(self):
        self.min: float = 0.0
        self.max: float = 1.0
        self.data_min: Optional[np.ndarray] = None
        self.data_max: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        scale = (self.max - self.min) / (self.data_max - self.data_min)
        return self.min + (data - self.data_min) * scale

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)