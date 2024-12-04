from similarity_users_rg_version.classes.user import User


def calculate_users_similarity(user1: User, user2: User, threshold: int = None) -> float:
    same_movies = 0
    same_rating = 0

    for movie in user1.train_set[:threshold]:
        for movie_ in user2.train_set:

            if movie.id == movie_.id:
                same_movies += 1

                if movie.rating == movie_.rating:
                    same_rating += 1
                    break

    return same_rating / same_movies


def assess_training_predictions(user1: User, user2: User, threshold: int = None):
    for movie in user1.train_set[threshold:]:
        for movie_ in user2.train_set:

            if movie.id == movie_.id:
                movie.predicted_rating = movie_.rating
                break


def assess_testing_predictions(user1: User, user2: User):
    for movie in user1.test_set:
        for movie_ in user2.train_set:

            if movie.id == movie_.id:
                movie.rating = movie_.rating
                break


def get_accuracy(user: User, threshold: int = None):
    same_movies = 0
    same_rating = 0

    for movie in user.train_set[threshold:]:
        same_movies += 1
        if movie.rating == movie.predicted_rating:
            same_rating += 1

    return same_rating / same_movies
