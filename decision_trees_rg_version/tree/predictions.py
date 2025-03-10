import numpy as np

from decision_trees_rg_version.classes.decision_tree import DecisionTree
from decision_trees_rg_version.classes.user import User
from decision_trees_rg_version.tree.data_adjuster import prepare_data_for_training, prepare_data_for_testing
from decision_trees_rg_version.utils.data_sorter import divide_list


def predict_with_tree(users: list[User], users_number: int = None):
    tree_hyperparams = {
        'max_depth': [2, 3, 4, 5, 6, 7],
        'min_samples_split': [2, 3, 4, 5, 6, 7],
    }

    for user in users[:users_number]:
        print(f"Processing user {user.user_id}...")

        train_data = user.train_movies
        data_slices = divide_list(train_data)

        best_accuracy = -float('inf')
        best_deep = -float('inf')
        best_min_samples = -float('inf')
        tree = None

        for depth in tree_hyperparams['max_depth']:
            for min_samples_split in tree_hyperparams['min_samples_split']:

                accuracies = []

                for i, test_slice in enumerate(data_slices):
                    test_set = test_slice
                    train_set = [item for j, slice in enumerate(data_slices) if j != i for item in slice]

                    X, y = prepare_data_for_training(train_set)
                    Z = prepare_data_for_testing(test_set)

                    tree = DecisionTree(max_depth=depth, min_samples_split=min_samples_split)
                    tree.fit(X, y)

                    predictions = tree.predict(Z)

                    true_ratings = [movie.rating for movie in test_set]

                    fold_accuracy = calculate_accuracy(predictions, true_ratings)
                    accuracies.append(fold_accuracy)

                avg_accuracy = np.mean(accuracies)

                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_deep = depth
                    best_min_samples = min_samples_split
                    user.set_tree(tree)

        print(
            f"For user {user.user_id}, the best depth is {best_deep}, and the best min_samples is {best_min_samples}, "
            f"accuracy for this is {best_accuracy * 100:.1f}%")

        user.predict_ratings_with_tree()


def calculate_accuracy(predictions, true_ratings):
    correct_predictions = sum([1 for pred, true in zip(predictions, true_ratings) if pred == true])
    accuracy = correct_predictions / len(true_ratings)
    return accuracy
