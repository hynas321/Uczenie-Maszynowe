from typing import List, Tuple, Any, Optional
import numpy as np
from utils.similarity_functions import compute_similarity

from collections import Counter

class KNN:
    def __init__(self, feature_types: List[Tuple[str, Any]], k: int = 5,
                 similarity_function=compute_similarity) -> None:
        self.k: int = k
        self.similarity_function: Any = similarity_function
        self.features: Optional[List[np.ndarray]] = None
        self.labels: Optional[List[int]] = None
        self.feature_types: List[Tuple[str, Any]] = feature_types

    def fit(self, features: List[np.ndarray], labels: List[int]) -> None:
        self.features = features
        self.labels = labels

    def predict(self, task_vector: np.ndarray) -> int:
        if self.features is None or self.labels is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' with training data.")

        similarities: List[Tuple[float, int]] = [
            (self.similarity_function(task_vector, feature_vector, self.feature_types), i)
            for i, feature_vector in enumerate(self.features)
        ]

        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = [self.labels[i] for _, i in similarities[:self.k]]

        most_common_label = Counter(top_k).most_common(1)[0][0]
        return np.clip(most_common_label, 0, 5)