from datetime import datetime

from decision_trees_rg_version.classes.movie import Movie


def get_adult_similarity(adu_1, adu_2):
    if adu_1 == adu_2:
        return 0
    else:
        return 1


def get_popularity_similarity(pop_1, pop_2):
    return float(abs(pop_1 - pop_2) / 100)


def get_vote_average_similarity(vot_1, vot_2):
    return float(abs(vot_1 - vot_2) / 10)


def get_vote_count_similarity(vot_1, vot_2):
    return float(abs(vot_1 - vot_2) / 500_000)


def get_budget_similarity(bud_1, bud_2):
    return float(abs(bud_1 - bud_2) / 1_000_000_000)


def get_revenue_similarity(rev_1, rev_2):
    return float(abs(rev_1 - rev_2) / 3_000_000_000)


def get_runtime_similarity(tim_1, tim_2):
    return float(abs(tim_1 - tim_2) / 300)


def get_release_date_similarity(dat_1, dat_2):
    date_format = "%Y-%m-%d"
    date1 = datetime.strptime(dat_1, date_format)
    date2 = datetime.strptime(dat_2, date_format)

    day_diff = abs((date2 - date1).days)

    max_days = 125 * 365
    return day_diff / max_days


def get_list_data_similarity(lis_1, lis_2):
    """
        Calculate the dissimilarity between two lists of languages.
        Lower overlap means higher dissimilarity.
        """
    # Convert lists to sets
    set1, set2 = set(lis_1), set(lis_2)

    # Calculate intersection and union
    intersection = set1 & set2  # Common elements
    union = set1 | set2  # All unique elements

    # Jaccard dissimilarity = 1 - |intersection| / |union|
    if not union:  # Avoid division by zero if both sets are empty
        return 0.0

    dissimilarity = 1 - (len(intersection) / len(union))
    return dissimilarity


def get_movies_similarity(mov_1: Movie, mov_2: Movie):
    d_1 = mov_1.movie_details
    d_2 = mov_2.movie_details

    score = get_adult_similarity(d_1.adult, d_2.adult) + get_popularity_similarity(d_1.popularity, d_2.popularity) + \
        get_vote_average_similarity(d_1.vote_average, d_2.vote_average) + \
        get_vote_count_similarity(d_1.vote_count, d_2.vote_count) + get_budget_similarity(d_1.budget, d_2.budget) + \
        get_revenue_similarity(d_1.revenue, d_2.revenue) + get_runtime_similarity(d_1.runtime, d_2.runtime) + \
        get_release_date_similarity(d_1.release_date, d_2.release_date) + \
        get_list_data_similarity(d_1.spoken_languages, d_2.spoken_languages) + \
        get_list_data_similarity(d_1.production_companies, d_2.production_companies) + \
        get_list_data_similarity(d_1.production_countries, d_2.production_countries)

    return score
