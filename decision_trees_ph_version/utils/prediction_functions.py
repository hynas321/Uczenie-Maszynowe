from collections import defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from tqdm import tqdm
from itertools import product
import csv

from knn.decision_tree import DecisionTree
from knn.random_forest import RandomForest


def predict_ratings(train_data_df: DataFrame, task_data_df: DataFrame,
                    movie_feature_vectors: Dict[int, np.ndarray], hyperparameter_values: Dict[str, List[Any]],
                    model_type: str, report_filename: str, feature_names: List[str]) -> Dict[int, int]:
    """
    Predicts ratings for user-movie pairs using Decision Tree or Random Forest
    """
    user_ratings = group_user_ratings(train_data_df)

    predictions = {}
    hyperparameter_scores = defaultdict(list)
    best_hyperparam_counts = defaultdict(int)

    arbitrary_user_id = None
    arbitrary_user_model = None

    for user_id in tqdm(task_data_df['user_id'].unique(), desc="Processing users"):
        X_train, y_train = prepare_user_training_data(user_id, user_ratings, movie_feature_vectors)

        if len(X_train) >= 2:
            best_params = select_best_params(X_train, y_train, hyperparameter_values, model_type, hyperparameter_scores,
                                             best_hyperparam_counts)

            if model_type == 'tree':
                model = DecisionTree(**best_params)
            elif model_type == 'forest':
                model = RandomForest(**best_params)
            else:
                raise ValueError("Invalid model type")

            model.fit(np.array(X_train), np.array(y_train))

            if arbitrary_user_id is None:
                arbitrary_user_id = user_id
                arbitrary_user_model = model

            predict_for_user(user_id, task_data_df, movie_feature_vectors, model, predictions, train_data_df)
        else:
            assign_default_ratings(user_id, task_data_df, predictions, train_data_df)

    save_hyperparameter_report(hyperparameter_scores, best_hyperparam_counts, report_filename)

    if arbitrary_user_model and model_type == 'tree':
        save_tree_visualization(arbitrary_user_model, 'tree.png', feature_names)

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


def select_best_params(X_train: List[np.ndarray], y_train: List[int], hyperparameter_values: Dict[str, List[Any]],
                       model_type: str, hyperparameter_scores: defaultdict, best_hyperparam_counts: defaultdict) -> \
Dict[str, Any]:
    """
    Selects the best hyperparameters using cross-validation.
    """

    hyperparam_names = list(hyperparameter_values.keys())
    hyperparam_combinations = list(product(*hyperparameter_values.values()))

    param_accuracies = {}

    for combination in hyperparam_combinations:
        params = dict(zip(hyperparam_names, combination))
        accuracy = cross_validate_params(X_train, y_train, params, model_type)

        param_key = tuple(sorted(params.items()))
        param_accuracies[param_key] = accuracy
        hyperparameter_scores[param_key].append(accuracy)

    # Select the best hyperparameters
    best_params_key = max(param_accuracies, key=param_accuracies.get)
    best_params = dict(best_params_key)
    best_hyperparam_counts[best_params_key] += 1

    return best_params


def cross_validate_params(X_train: List[np.ndarray], y_train: List[int], params: Dict[str, Any], model_type: str,
                          test_size: float = 0.2) -> float:
    """
    Cross-validates hyperparameters to calculate the accuracy.
    """
    n_samples = len(X_train)
    if n_samples < 5:
        return 0.0

    test_indices = np.random.choice(range(n_samples), size=int(n_samples * test_size), replace=False)
    train_indices = list(set(range(n_samples)) - set(test_indices))

    X_fold_train = [X_train[i] for i in train_indices]
    y_fold_train = [y_train[i] for i in train_indices]
    X_fold_test = [X_train[i] for i in test_indices]
    y_fold_test = [y_train[i] for i in test_indices]

    if model_type == 'tree':
        model = DecisionTree(**params)
    elif model_type == 'forest':
        model = RandomForest(**params)
    else:
        raise ValueError("Invalid model_type. Must be 'tree' or 'forest'.")

    model.fit(np.array(X_fold_train), np.array(y_fold_train))
    predictions = model.predict(np.array(X_fold_test))

    accuracy = np.mean([1 if pred == true else 0 for pred, true in zip(predictions, y_fold_test)])
    return accuracy


def predict_for_user(user_id: int, task_data_df: DataFrame, movie_feature_vectors: Dict[int, np.ndarray],
                     model, predictions: Dict[int, int], train_data_df: DataFrame) -> None:
    """
    Predicts ratings for a user and updates the predictions dictionary.
    """
    user_task_indices = task_data_df[task_data_df['user_id'] == user_id].index

    for idx in user_task_indices:
        movie_id = task_data_df.loc[idx, 'movie_id']
        if movie_id not in movie_feature_vectors:
            predictions[idx] = int(round(train_data_df['rating'].mean()))  # Default rating
            continue

        task_vector = movie_feature_vectors[movie_id]
        predicted_rating = model.predict(task_vector.reshape(1, -1))[0]
        predictions[idx] = int(round(predicted_rating))


def assign_default_ratings(user_id: int, task_data_df: DataFrame, predictions: Dict[int, int],
                           train_data_df: DataFrame):
    """
    Assign default ratings for a user when not enough data is available.
    """
    user_task_indices = task_data_df[task_data_df['user_id'] == user_id].index
    default_rating = int(round(train_data_df['rating'].mean()))
    for idx in user_task_indices:
        predictions[idx] = default_rating


def save_hyperparameter_report(hyperparameter_scores: Dict[Tuple, List[float]],
                               best_hyperparam_counts: Dict[Tuple, int],
                               report_filename: str) -> None:
    with open(report_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Hyperparameters", "Count of Best", "Average Accuracy"])

        for params_key in sorted(hyperparameter_scores.keys()):
            accuracies = hyperparameter_scores[params_key]
            average_accuracy = np.mean(accuracies) if accuracies else 0
            count = best_hyperparam_counts.get(params_key, 0)
            params_str = ', '.join(f'{k}={v}' for k, v in sorted(dict(params_key).items()))
            writer.writerow([params_str, count, average_accuracy])

    print(f"Parameter selection report saved to {report_filename}")


def save_tree_visualization(model: DecisionTree, filename: str, feature_names: List[str]) -> None:
    def plot_node(ax, text, center, parent_center=None):
        bbox_props = dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", lw=1)
        ax.text(center[0], center[1], text, ha="center", va="center", bbox=bbox_props)
        if parent_center:
            ax.plot([parent_center[0], center[0]], [parent_center[1], center[1]], 'k-')

    def traverse_and_plot(node, ax, x=0.5, y=1.0, dx=0.1, dy=0.1, parent_center=None):
        if node.is_leaf_node():
            text = f"Value: {node.value}"
        else:
            feature_name = feature_names[node.feature]
            text = f"{feature_name} <= {node.threshold:.2f}"

        current_center = (x, y)
        plot_node(ax, text, current_center, parent_center)

        if not node.is_leaf_node():
            # Left
            traverse_and_plot(node.left, ax, x - dx, y - dy, dx / 2, dy, current_center)
            # Right
            traverse_and_plot(node.right, ax, x + dx, y - dy, dx / 2, dy, current_center)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    traverse_and_plot(model.root, ax)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Decision tree visualization saved to {filename}")
