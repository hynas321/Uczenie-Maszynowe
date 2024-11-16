import json
from typing import Any

from rg_version.classes.movie import Movie


class User:
    def __init__(self, user_id: int, train_movies: list[Movie], test_movies: list[Movie]):
        self.user_id: int = user_id
        self.train_movies: list[Movie] = train_movies
        self.test_movies: list[Movie] = test_movies
        self.k: int | None = None

    def set_k_parameter(self, k):
        self.k = k

    def to_dict(self):
        return {
            "user_id": str(self.user_id),
            "k": str(self.k),
        }

    def __repr__(self) -> str:
        return json.dumps(self.to_dict())
