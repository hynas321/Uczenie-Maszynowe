from typing import List

import numpy as np

from decision_trees_rg_version.classes.movie import Movie
from decision_trees_rg_version.classes.decision_tree import DecisionTree
from decision_trees_rg_version.classes.random_forest import RandomForest
from decision_trees_rg_version.tree.data_adjuster import preprocess_movie_details


class User:
    def __init__(self, user_id: int, train_movies: List[Movie], test_movies: List[Movie]):
        self.user_id: int = user_id
        self.train_movies: List[Movie] = train_movies
        self.test_movies: List[Movie] = test_movies
        self.tree: DecisionTree = None
        self.forest: RandomForest = None

    def set_tree(self, tree: DecisionTree):
        self.tree = tree

    def set_forest(self, forest: RandomForest):
        self.forest = forest

    def predict_ratings_with_tree(self):
        for movie in self.test_movies:
            movie_features = preprocess_movie_details(movie.movie_details)
            movie.rating = int(self.tree.predict(np.array(movie_features).reshape(1, -1))[0])

    def predict_ratings_with_forest(self):
        for movie in self.test_movies:
            movie_features = preprocess_movie_details(movie.movie_details)
            movie.rating = int(self.forest.predict(np.array(movie_features).reshape(1, -1))[0])

    def print_prediction(self):
        predictions = []
        for movie in self.test_movies:
            predictions.append(movie.rating)

        return predictions
