from typing import Optional

import numpy as np

class MinMaxScaler:
    def __init__(self):
        self.min_: float = 0.0
        self.max_: float = 1.0
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        self.data_min_ = np.min(data, axis=0)
        self.data_max_ = np.max(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        scale = (self.max_ - self.min_) / (self.data_max_ - self.data_min_)
        return self.min_ + (data - self.data_min_) * scale

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)