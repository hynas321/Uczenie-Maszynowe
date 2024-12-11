import time

from collaborative_filtering_users_rg_version.utils.csv_saver import save_data_to_csv
from collaborative_filtering_users_rg_version.utils.data_fetch import get_users


# Funkcja przewidująca ocenę
def predict_rating(p_r, x_m):
    prediction = sum(p * x for p, x in zip(p_r, x_m)) + p_r[-1]  # Ostatni element to bias p_0
    # Ograniczanie wartości oceny do przedziału 0-5
    return max(0, min(5, round(prediction)))  # Zaokrąglenie i ograniczenie



# Funkcja gradientu dla parametrów użytkownika
def compute_gradients(p_r, x_m_list, y_list):
    gradients = [0] * len(p_r)
    for x_m, y in zip(x_m_list, y_list):
        error = predict_rating(p_r, x_m) - y
        gradients[-1] += error  # Gradient dla biasu p_0
        for i in range(len(x_m)):
            gradients[i] += error * x_m[i]  # Gradient dla p_i
    return gradients


# Aktualizacja parametrów gradientu
def update_parameters(p_r, gradients, eta):
    for i in range(len(p_r)):
        p_r[i] -= eta * gradients[i]


# Główna pętla uczenia
def train_user_parameters(x_m_list, y_list, n_features, eta, epochs):
    p_r = [0] * (n_features + 1)  # Inicjalizacja parametrów użytkownika (n cech + bias)
    for epoch in range(epochs):
        gradients = compute_gradients(p_r, x_m_list, y_list)
        update_parameters(p_r, gradients, eta)
    return p_r


def main():
    start_time = time.time()

    # getting users data
    users = get_users()
    print("Data loaded...")

    # Przykład użycia
    for user in users:
        train_set = user.train_set  # Filmy ocenione przez użytkownika
        x_m_list = [movie.get_features() for movie in train_set]  # Cechy filmów
        y_list = [movie.rating for movie in train_set]  # Oceny filmów
        user.p_r = train_user_parameters(x_m_list, y_list, n_features=9, eta=0.01, epochs=100)

        # Przewidywanie ocen dla test_set
        for movie in user.test_set:
            movie.rating = predict_rating(user.p_r, movie.get_features())

        print(user.test_set)

    save_data_to_csv(users)

    print(f"Elapsed time: {(time.time() - start_time):.2f} seconds")


if __name__ == '__main__':
    main()
