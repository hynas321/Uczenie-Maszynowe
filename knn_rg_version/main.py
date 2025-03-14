import time

from knn_rg_version.knn.knn import calculate_best_k_for_user, calculate_predictions
from knn_rg_version.utils.csv_saver import save_data_to_csv
from knn_rg_version.utils.data_fetch import get_users


def main() -> None:
    start_time = time.time()

    users = get_users()

    users_with_k = calculate_best_k_for_user(users)

    users_with_predictions = calculate_predictions(users_with_k)

    save_data_to_csv(users_with_predictions)

    print(f"Elapsed time: {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    main()
