import numpy as np

from utils.similarity_functions import compute_similarity

class KNN:
    def __init__(self, feature_types, k=5, similarity_function=compute_similarity):
        self.k = k
        self.similarity_function = similarity_function
        self.features = None
        self.labels = None
        self.feature_types = feature_types

    def fit(self, features, labels):
        self.features = features
        self.labels = labels

    def predict(self, task_vector):
        if self.features is None or self.labels is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' with training data.")

        similarities = []
        for idx in range(len(self.features)):
            feature_vector = self.features[idx]
            similarity = self.similarity_function(task_vector, feature_vector, self.feature_types)
            similarities.append((similarity, idx))

        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = similarities[:self.k]

        numerator = 0.0
        denominator = 0.0
        for similarity, idx in top_k:
            label = self.labels[idx]
            numerator += similarity * label
            denominator += similarity

        if denominator == 0:
            return int(round(np.mean(self.labels)))

        predicted_label = numerator / denominator

        return max(0, min(5, int(round(predicted_label))))

    def predict_batch(self, task_vectors):
        predictions = []
        for task_vector in task_vectors:
            prediction = self.predict(task_vector)
            predictions.append(prediction)
        return predictions
