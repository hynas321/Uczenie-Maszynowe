import pandas as pd
import os

from dotenv import load_dotenv
from tmdb_api_service import TmdbApiService

movie_data = pd.read_csv('movie/movie.csv', delimiter=';', index_col=0, usecols=[0, 1, 2], names=["movie_id", "tmbd_movie_id", "title"])
task_data = pd.read_csv('movie/task.csv', delimiter=';', index_col=0, usecols=[0, 1, 2, 3], names=["index", "user_id", "movie_id", "rating"])
train_data = pd.read_csv('movie/train.csv', delimiter=';', index_col=0, usecols=[0, 1, 2, 3], names=["index", "user_id", "movie_id", "rating"])

load_dotenv()

api_key = os.getenv('TMDB_API_KEY')
tmdb_api_service = TmdbApiService(api_key)

movie_ids = [389, 62]

#Sample movie details fetching
for movie_id in movie_ids:
    details = tmdb_api_service.fetch_movie_details(movie_id)
    print(f"Title: {details.title}")
    print(f"Release Date: {details.release_date}")
    print(f"Runtime: {details.runtime} minutes")
    print(f"Genres: {', '.join([genre['name'] for genre in details.genres])}")
    print(f"Rating: {details.vote_average}/10")
    print("-" * 40)