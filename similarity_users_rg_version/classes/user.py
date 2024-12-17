from similarity_users_rg_version.classes.movie import Movie


class User:
    def __init__(self, user_id: int, train_set: list[Movie] = [], test_set: list[Movie] = []):
        self.id = user_id
        self.train_set: list[Movie] = train_set
        self.test_set: list[Movie] = test_set

    def __str__(self):
        return f'User with id: {self.id} and {len(self.train_set)} train, {len(self.test_set)} test movies.'

    def set_train_set(self, train_set: list[Movie]):
        self.train_set: list[Movie] = train_set

    def set_test_set(self, test_set: list[Movie]):
        self.test_set: list[Movie] = test_set

    def show_train_set(self):
        for movie in self.train_set[72:]:
            print(movie)
