from similarity_users_rg_version.classes.user import User
from similarity_users_rg_version.utils.similarity import calculate_users_similarity, get_accuracy, \
    assess_training_predictions


def cross_validate(users: list[User], threshold: int = None):
    all_accuracy = []

    for user in users:

        similarities_with_user = []

        for user_ in users:
            users_similarity = calculate_users_similarity(user, user_, threshold)
            similarities_with_user.append([users_similarity, user_])

        similarities_with_user.sort(key=lambda x: x[0])
        for similarity, user_ in similarities_with_user:
            print(f"{similarity}\t{user_}")

        users_table = [user_[1] for user_ in similarities_with_user]
        users_table = users_table[:len(users_table) - 1]

        for user__ in users_table:
            assess_training_predictions(user, user__, threshold)

        # user.show_train_set()
        accuracy = 100 * get_accuracy(user, threshold)

        all_accuracy.append(accuracy)
        print(f'For user with id: {user.id}, the accuracy is: {accuracy:.1f} %')

    print(f'Overall accuracy: {(sum(all_accuracy) / len(all_accuracy)):.1f} %')