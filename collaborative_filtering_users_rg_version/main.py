import time
from itertools import product

from collaborative_filtering_users_rg_version.utils.csv_saver import save_data_to_csv
from collaborative_filtering_users_rg_version.utils.data_fetch import get_users
from collaborative_filtering_users_rg_version.utils.data_sorter import divide_list


def predict_rating(p_r, x_m):
    prediction = sum(p * x for p, x in zip(p_r, x_m)) + p_r[-1]
    return max(0, min(5, round(prediction)))


def compute_gradients(p_r, x_m_list, y_list):
    gradients = [0] * len(p_r)
    for x_m, y in zip(x_m_list, y_list):
        error = predict_rating(p_r, x_m) - y
        gradients[-1] += error
        for i in range(len(x_m)):
            gradients[i] += error * x_m[i]
    return gradients


def update_parameters(p_r, gradients, eta):
    for i in range(len(p_r)):
        p_r[i] -= eta * gradients[i]


def train_user_parameters(x_m_list, y_list, n_features, eta, epochs):
    p_r = [0] * (n_features + 1)
    for epoch in range(epochs):
        gradients = compute_gradients(p_r, x_m_list, y_list)
        update_parameters(p_r, gradients, eta)
    return p_r


def main():
    start_time = time.time()

    users = get_users()
    print("Data loaded...")

    epochs = [50, 100, 200, 300, 400, 500]
    etas = [0.001, 0.005, 0.01, 0.05, 0.1]

    for a, user in enumerate(users):
        best_eta, best_epoch = 0, 0
        best_accuracy = 0

        train_data = user.train_set
        test_data = user.test_set
        data_slices = divide_list(train_data)

        for eta, epoch in product(etas, epochs):
            accuracy = 0

            for i, test_slice in enumerate(data_slices):
                proper_rating_proposal = 0

                test_set = test_slice
                train_set = [item for j, slice_ in enumerate(data_slices) if j != i for item in slice_]

                x_m_list = [movie.get_features() for movie in train_set]
                y_list = [movie.rating for movie in train_set]

                user.p_r = train_user_parameters(x_m_list, y_list, n_features=9, eta=eta, epochs=epoch)

                for movie in test_set:
                    if movie.rating == predict_rating(user.p_r, movie.get_features()):
                        proper_rating_proposal += 1

                accuracy += proper_rating_proposal/len(test_set)

            accuracy /= 5

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_eta = eta
                best_epoch = epoch

        print(f'{a + 1}. Best accuracy: {100 * best_accuracy:.2f}%, eta: {best_eta}, epoch: {best_epoch}')

        x_m_list = [movie.get_features() for movie in train_data]
        y_list = [movie.rating for movie in train_data]

        user.p_r = train_user_parameters(x_m_list, y_list, n_features=9, eta=best_eta, epochs=best_epoch)
        for movie in test_data:
            movie.rating = predict_rating(user.p_r, movie.get_features())

    save_data_to_csv(users)
    print(f"Elapsed time: {(time.time() - start_time):.2f} seconds")


if __name__ == '__main__':
    main()
