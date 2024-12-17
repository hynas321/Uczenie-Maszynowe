
class Movie:
    def __init__(self, movie_id: int, rating: int = None, ):
        self.id = movie_id
        self.rating = rating
        self.predicted_rating = None

    def set_predicted_rating(self, predicted_rating):
        self.predicted_rating = predicted_rating

    def __str__(self):
        return f'Movie with id: {self.id} and rating: {self.rating} and predicted_rating: {self.predicted_rating}'

    def is_predicted_rating_same_as_rating(self):
        return int(self.predicted_rating == self.rating)
