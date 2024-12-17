from similarity_users_rg_version.similarity.cross_validation import cross_validate
from similarity_users_rg_version.similarity.predict_values import predict_values
from similarity_users_rg_version.utils.csv_reader import load_csv_data
from similarity_users_rg_version.utils.csv_saver import save_data_to_csv
from similarity_users_rg_version.utils.users_creation import generate_users


def main():
    train_data, test_data = load_csv_data()
    users = generate_users(train_data=train_data, task_data=test_data)

    # cross_validate(users=users, threshold=72)

    predict_values(users=users)

    save_data_to_csv(users=users)

if __name__ == '__main__':
    main()
