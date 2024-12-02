import time

from decision_trees_rg_version.forest.predictions import predict_with_forest
from decision_trees_rg_version.tree.predictions import predict_with_tree
from decision_trees_rg_version.utils.csv_saver import save_data_to_csv
from decision_trees_rg_version.utils.data_fetch import get_users


def main() -> None:
    start_time = time.time()

    # getting users data
    users_tree = get_users()
    users_forest = users_tree
    print("Data loaded...")

    # getting users prediction with Decision Tree with best hyperparameters
    predict_with_tree(users_tree)
    save_data_to_csv(users_tree)

    # getting users prediction with Random Forest with the best hyperparameters
    predict_with_forest(users_forest)
    save_data_to_csv(users_forest)

    print(f"Elapsed time: {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    main()
