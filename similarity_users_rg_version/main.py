from similarity_users_rg_version.utils.csv_reader import load_csv_data
from similarity_users_rg_version.utils.similarity import calculate_users_similarity, assess_predictions, get_accuracy
from similarity_users_rg_version.utils.users_creation import generate_users


def main():
    train_data, test_data = load_csv_data()
    users = generate_users(train_data, test_data)

    all_accuracy = []

    for user in users:

        similarities_with_user = []

        for user_ in users:
            users_similarity = calculate_users_similarity(user, user_)
            similarities_with_user.append([users_similarity, user_])

        similarities_with_user.sort(key=lambda x: x[0])
        users_table = [user_[1] for user_ in similarities_with_user]
        users_table = users_table[:len(users_table) - 1]

        for user__ in users_table:
            assess_predictions(user, user__)

        # user.show_train_set()
        accuracy = 100*get_accuracy(user)

        all_accuracy.append(accuracy)
        print(f'For user with id: {user.id}, the accuracy is: {accuracy:.1f} %')

    print(f'Overall accuracy: {(sum(all_accuracy) / len(all_accuracy)):.1f} %')

if __name__ == '__main__':
    main()
