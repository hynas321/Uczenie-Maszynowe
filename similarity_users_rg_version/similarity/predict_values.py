from similarity_users_rg_version.classes.user import User
from similarity_users_rg_version.utils.similarity import calculate_users_similarity, assess_testing_predictions, \
    get_accuracy


def predict_values(users: list[User]):

    for user in users:

        similarities_with_user = []

        for user_ in users:
            users_similarity = calculate_users_similarity(user, user_)
            similarities_with_user.append([users_similarity, user_])

        similarities_with_user.sort(key=lambda x: x[0])

        users_table = [user_[1] for user_ in similarities_with_user]
        users_table = users_table[:len(users_table) - 1]

        for user__ in users_table:
            assess_testing_predictions(user, user__)

