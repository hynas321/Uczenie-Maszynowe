from typing import List, Tuple, Any, Optional
import numpy as np
from utils.similarity_functions import compute_similarity


class KNN:
    def __init__(self, feature_types: List[Tuple[str, Any]], k: int = 5, similarity_function=compute_similarity) -> None:
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
        top_k = similarities[:self.k]

        numerator = sum(similarity * self.labels[i] for similarity, i in top_k)
        denominator = sum(similarity for similarity, _ in top_k)

        if denominator == 0:
            return int(round(np.mean(self.labels)))

        predicted_label = numerator / denominator
        return np.clip(int(round(predicted_label)), 0, 5)

    def cross_validate(self, features: List[np.ndarray], labels: List[int], k_folds: int = 5):
        n_samples = len(features)
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)

        fold_size = n_samples // k_folds
        scores = []

        for i in range(k_folds):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

            X_train = [features[j] for j in train_indices]
            y_train = [labels[j] for j in train_indices]
            X_test = [features[j] for j in test_indices]
            y_test = [labels[j] for j in test_indices]

            self.fit(X_train, y_train)
            predictions = [self.predict(x) for x in X_test]
            accuracy = np.mean([1 if pred == true else 0 for pred, true in zip(predictions, y_test)])
            scores.append(accuracy)

        return np.mean(scores)