from knn_rg_version.classes.user import User
from knn_rg_version.utils.data_sorter import divide_list, get_k_best_point
from knn_rg_version.utils.similiarity_functions import get_movies_similarity


def calculate_best_k_for_user(users: list[User], user_range: int = 10):
    k_values = range(1, 2, 2)

    for user in users[:user_range]:
        train_data = user.train_movies

        data_slices = divide_list(train_data)

        best_accuracy = 0

        for k in k_values:

            accuracy = 0

            for i, test_slice in enumerate(data_slices):

                test_set = test_slice
                train_set = [item for j, slice in enumerate(data_slices) if j != i for item in slice]

                proper_rating_proposal = 0

                for test_point in test_set:

                    points = []

                    for train_point in train_set:
                        score = get_movies_similarity(test_point, train_point)
                        points.append([score, train_point.rating])

                    most_frequent_point_rating = get_k_best_point(points, k)

                    # print(f"Calculated rating = {most_frequent}, but the real rating = {test_point.rating}")
                    if most_frequent_point_rating == test_point.rating:
                        proper_rating_proposal += 1

                accuracy += (proper_rating_proposal / len(test_set))

            k_accuracy = accuracy / 5

            if k_accuracy > best_accuracy:
                user.k = k
                best_accuracy = k_accuracy

        # print(f"For user with id = {user.user_id}, the best accuracy = {(best_accuracy * 100):.1f} %,
    print("Best k for each user found...")

    return users[:user_range]


def calculate_predictions(users: list[User]):
    for user in users:

        k = user.k
        train_dataset = user.train_movies
        task_dataset = user.test_movies

        for task_movie in task_dataset:

            task_list = []

            for train_movie in train_dataset:
                score = get_movies_similarity(task_movie, train_movie)
                task_list.append([score, train_movie.rating])

            most_frequent_point_rating = get_k_best_point(task_list, k)
            task_movie.rating = most_frequent_point_rating

        user.test_movies = task_dataset
    print("Predictions for each user calculated...")

    return users
