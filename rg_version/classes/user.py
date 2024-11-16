
class User:
    def __init__(self, user_id, train_movies, test_movies):
        self.user_id = user_id
        self.train_movies = train_movies
        self.test_movies = test_movies
        self.k = None

    def set_k_parameter(self, k):
        self.k = k
