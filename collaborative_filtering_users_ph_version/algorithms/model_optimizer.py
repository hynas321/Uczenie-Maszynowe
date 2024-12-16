from itertools import product
from typing import List, Tuple


class ModelOptimizer:
    def __init__(self, learning_rates: List[float], num_epochs: List[int], num_features: int):
        self.learning_rates = learning_rates
        self.num_epochs = num_epochs
        self.num_features = num_features

    def calculate_prediction(self, user_params: List[float], feature_vector: List[float]) -> float:
        rating = sum(param * feature for param, feature in zip(user_params, feature_vector)) + user_params[-1]
        return max(0, min(5, round(rating)))

    def compute_error_gradients(self, user_params: List[float], feature_matrix: List[List[float]],
                                actual_ratings: List[int], lambda_reg: float = 0.01) -> List[float]:
        gradients = [0] * len(user_params)
        for feature_vec, actual_rating in zip(feature_matrix, actual_ratings):
            error = self.calculate_prediction(user_params, feature_vec) - actual_rating
            gradients[-1] += error
            for idx in range(len(feature_vec)):
                gradients[idx] += error * feature_vec[idx]

        for idx in range(len(user_params) - 1):
            gradients[idx] += lambda_reg * user_params[idx]

        return gradients

    def adjust_parameters(self, user_params: List[float], gradients: List[float], learning_rate: float,
                          lambda_reg: float = 0.01):
        for idx in range(len(user_params)):
            if idx < len(user_params) - 1:
                user_params[idx] -= learning_rate * (gradients[idx] + lambda_reg * user_params[idx])
            else:
                user_params[idx] -= learning_rate * gradients[idx]

    def optimize_user_params(self, feature_matrix: List[List[float]], actual_ratings: List[int], learning_rate: float,
                             epochs: int, lambda_reg: float = 0.01) -> List[float]:
        user_params = [0] * (self.num_features + 1)

        for _ in range(epochs):
            gradients = self.compute_error_gradients(user_params, feature_matrix, actual_ratings, lambda_reg)
            self.adjust_parameters(user_params, gradients, learning_rate, lambda_reg)

        return user_params

    def find_optimal_hyperparameters(self, data_splits: List[List[Tuple[List[float], int]]]) -> Tuple[float, int, float]:
        optimal_lr, optimal_epoch, highest_accuracy = 0, 0, 0
        for learning_rate, epoch_count in product(self.learning_rates, self.num_epochs):
            avg_accuracy = 0
            for idx, validation_split in enumerate(data_splits):
                training_data = []
                for split_idx, split in enumerate(data_splits):
                    if split_idx != idx:
                        training_data.extend(split)

                feature_matrix = []
                actual_ratings = []
                for features, rating in training_data:
                    feature_matrix.append(features)
                    actual_ratings.append(rating)

                user_params = self.optimize_user_params(feature_matrix, actual_ratings, learning_rate, epoch_count)

                valid_predictions = 0
                for features, actual_rating in validation_split:
                    prediction = self.calculate_prediction(user_params, features)
                    if round(prediction) == actual_rating:
                        valid_predictions += 1

                avg_accuracy += valid_predictions / len(validation_split)

            avg_accuracy /= len(data_splits)
            if avg_accuracy > highest_accuracy:
                optimal_lr, optimal_epoch, highest_accuracy = learning_rate, epoch_count, avg_accuracy

        return optimal_lr, optimal_epoch, highest_accuracy
