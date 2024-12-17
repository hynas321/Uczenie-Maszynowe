import numpy as np

from decision_trees_rg_version.classes.decision_tree import DecisionTree
from decision_trees_rg_version.classes.random_forest import RandomForest
from decision_trees_rg_version.classes.user import User
from decision_trees_rg_version.tree.data_adjuster import prepare_data_for_training, prepare_data_for_testing
from decision_trees_rg_version.utils.data_sorter import divide_list


def predict_with_forest(users: list[User], users_number: int = None):

    tree_hyperparams = {
        'n_estimators': [10, 20, 30, 50, 70, 100],
    }

    for a, user in enumerate(users[:users_number]):
        print(f"Processing user {user.user_id}...")

        train_data = user.train_movies
        test_data = user.test_movies
        data_slices = divide_list(train_data)

        best_accuracy = -float('inf')
        best_tree_numbers = -float('inf')
        forest = None

        for tree_numbers in tree_hyperparams['n_estimators']:

            accuracies = []

            for i, test_slice in enumerate(data_slices):
                test_set = test_slice
                train_set = [item for j, slice in enumerate(data_slices) if j != i for item in slice]

                X, y = prepare_data_for_training(train_set)
                Z = prepare_data_for_testing(test_set)

                forest = RandomForest(n_estimators=tree_numbers, max_depth=5, max_features=3)
                forest.fit(X, y)

                # Predict on the test data
                predictions = forest.predict(Z)

                # Get true ratings for the test set
                true_ratings = [movie.rating for movie in test_set]

                # Calculate accuracy (exact match rate)
                fold_accuracy = calculate_accuracy(predictions, true_ratings)
                accuracies.append(fold_accuracy)

                # Calculate the average accuracy for this hyperparameter setting
            avg_accuracy = np.mean(accuracies)

            # print(f"Hyperparameters: tree_numbers={tree_numbers},> Accuracy: {avg_accuracy*100:.1f}%")

            # Keep track of the best tree based on accuracy
            if avg_accuracy > best_accuracy:  # Higher accuracy is better
                best_accuracy = avg_accuracy
                best_tree_numbers = tree_numbers
                user.set_forest(forest)

        print(
            f"{a+1}. For user {user.user_id}, the best tree number is {best_tree_numbers}, with accuracy = {best_accuracy * 100:.1f}%")

        user.predict_ratings_with_forest()


def calculate_accuracy(predictions, true_ratings):
    # Compare predictions with true ratings and calculate the accuracy
    correct_predictions = sum([1 for pred, true in zip(predictions, true_ratings) if pred == true])
    accuracy = correct_predictions / len(true_ratings)
    return accuracy
