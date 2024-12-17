from collections import defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np
from numpy import floating
from pandas import DataFrame
from tqdm import tqdm

from knn_ph_version.class_models.movie_features import MovieFeatures
from knn_ph_version.knn.knn import KNN
from knn_ph_version.utils.csv_functions import save_k_report


def predict_ratings(train_data_df: DataFrame, task_data_df: DataFrame,
                    movie_feature_vectors: Dict[int, np.ndarray], k_values: List[int],
                    report_filename: str = "knn_report.csv") -> Dict[int, int]:
    """
    Predicts ratings for user-movie pairs
    """
    user_ratings: dict[int, list[tuple[int, int]]] = group_user_ratings(train_data_df)

    predictions = {}
    k_scores = defaultdict(list)
    best_k_counts = defaultdict(int)

    for user_id in tqdm(task_data_df['user_id'].unique(), desc="Processing users"):
        X_train, y_train = prepare_user_training_data(user_id, user_ratings, movie_feature_vectors)

        best_k = select_best_k(X_train, y_train, k_values, k_scores, best_k_counts) if len(X_train) >= 2 else k_values[0]

        knn = KNN(feature_types=MovieFeatures.feature_types(), k=best_k)
        knn.fit(X_train, y_train)

        predict_for_user(user_id, task_data_df, movie_feature_vectors, knn, predictions, train_data_df)

    save_k_report(k_scores, best_k_counts, report_filename)

    return predictions

def group_user_ratings(train_data_df: DataFrame) -> Dict[int, List[Tuple[int, int]]]:
    """
    Groups user ratings from the training data.
    """
    user_ratings = defaultdict(list)
    for _, row in train_data_df.iterrows():
        user_ratings[row['user_id']].append((row['movie_id'], row['rating']))
    return user_ratings

def prepare_user_training_data(user_id: int, user_ratings: Dict[int, List[Tuple[int, int]]],
                               movie_feature_vectors: Dict[int, np.ndarray]) -> Tuple[List[np.ndarray], List[int]]:
    """
    Prepares training data for a specific user.
    """
    user_movies = user_ratings.get(user_id, [])
    X_train = [movie_feature_vectors[movie] for movie, _ in user_movies if movie in movie_feature_vectors]
    y_train = [rating for movie, rating in user_movies if movie in movie_feature_vectors]
    return X_train, y_train

def select_best_k(X_train: List[np.ndarray], y_train: List[int], k_values: List[int],
                  k_scores: defaultdict, best_k_counts: defaultdict) -> int:
    """
    Selects the best k-value using cross-validation.
    """
    k_accuracies = {}
    for k in k_values:
        accuracy = cross_validate_k(X_train, y_train, k)
        k_accuracies[k] = accuracy
        k_scores[k].append(accuracy)

    best_k = max(k_accuracies, key=k_accuracies.get)
    best_k_counts[best_k] += 1
    return best_k

def cross_validate_k(X_train: List[np.ndarray], y_train: List[int], k: int, test_size: float = 0.2) -> floating[Any]:
    """
    Cross-validates a k-value to calculate the accuracy.
    """
    n_samples = len(X_train)
    test_indices = np.random.choice(range(n_samples), size=int(n_samples * test_size), replace=False)
    train_indices = list(set(range(n_samples)) - set(test_indices))

    X_fold_train = [X_train[i] for i in train_indices]
    y_fold_train = [y_train[i] for i in train_indices]
    X_fold_test = [X_train[i] for i in test_indices]
    y_fold_test = [y_train[i] for i in test_indices]

    knn = KNN(feature_types=MovieFeatures.feature_types(), k=k)
    knn.fit(X_fold_train, y_fold_train)
    predictions = [knn.predict(x) for x in X_fold_test]

    accuracy = np.mean([1 if pred == true else 0 for pred, true in zip(predictions, y_fold_test)])
    return accuracy

def predict_for_user(user_id: int, task_data_df: DataFrame, movie_feature_vectors: Dict[int, np.ndarray],
                     knn: KNN, predictions: Dict[int, int], train_data_df: DataFrame) -> None:
    """
    Predicts ratings for a user and updates the predictions dictionary.
`   """
    user_task_indices = task_data_df[task_data_df['user_id'] == user_id].index

    for idx in user_task_indices:
        movie_id = task_data_df.loc[idx, 'movie_id']
        if movie_id not in movie_feature_vectors:
            predictions[idx] = 3  # Default rating
            continue

        task_vector = movie_feature_vectors[movie_id]
        if knn and knn.features:
            predicted_rating = knn.predict(task_vector)
            predictions[idx] = predicted_rating
        else:
            predictions[idx] = int(round(train_data_df['rating'].mean()))