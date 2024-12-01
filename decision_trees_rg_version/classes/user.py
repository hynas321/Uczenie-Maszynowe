from typing import List


from decision_trees_rg_version.classes.movie import Movie
from decision_trees_rg_version.classes.decision_tree import DecisionTree
from decision_trees_rg_version.classes.random_forest import RandomForest
from decision_trees_rg_version.tree.data_adjuster import preprocess_movie_details


class User:
    def __init__(self, user_id: int, train_movies: List[Movie], test_movies: List[Movie]):
        self.user_id: int = user_id
        self.train_movies: List[Movie] = train_movies
        self.test_movies: List[Movie] = test_movies
        self.tree: DecisionTree = None  # Tree parameter for the user
        self.forest: RandomForest = None

    def set_tree(self, tree: DecisionTree):
        """Assign a trained decision tree to the user."""
        self.tree = tree

    def set_forest(self, forest: RandomForest):
        """Assign a trained decision tree to the user."""
        self.forest = forest

    def predict_ratings_with_tree(self):
        """Predict ratings for the user's test movies."""

        for movie in self.test_movies:
            movie_features = preprocess_movie_details(movie.movie_details)
            movie.rating = self.tree.predict(movie_features)

    def predict_ratings_with_forest(self):
        """Predict ratings for the user's test movies."""

        for movie in self.test_movies:
            movie_features = preprocess_movie_details(movie.movie_details)
            movie.rating = self.forest.predict(movie_features)

    def print_prediction(self):
        predictions = []
        for movie in self.test_movies:
            predictions.append(movie.rating)

        return predictions
