
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors


def build_collaborative_model(ratings):

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    user_indices = user_encoder.fit_transform(
        ratings["userId"]
    )

    movie_indices = movie_encoder.fit_transform(
        ratings["movieId"]
    )

    user_item_matrix = csr_matrix(
        (
            ratings["rating"].values,
            (user_indices, movie_indices)
        ),
        shape=(
            len(user_encoder.classes_),
            len(movie_encoder.classes_)
        )
    )

    item_user_matrix = user_item_matrix.T.tocsr()

    collaborative_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=11,
        n_jobs=-1
    )

    collaborative_model.fit(item_user_matrix)

    return (
        user_encoder,
        movie_encoder,
        user_item_matrix,
        item_user_matrix,
        collaborative_model
    )


def get_collaborative_recommendations(
    user_id,
    ratings,
    movies,
    user_encoder,
    movie_encoder,
    user_item_matrix,
    item_user_matrix,
    collaborative_model,
    n=10
):

    if user_id not in user_encoder.classes_:
        return None

    user_index = user_encoder.transform([user_id])[0]

    user_ratings = user_item_matrix[user_index]

    rated_movie_indices = user_ratings.indices
    rated_movie_scores = user_ratings.data

    rated_movie_set = set(rated_movie_indices)

    scores = {}

    for movie_index, rating in zip(
        rated_movie_indices,
        rated_movie_scores
    ):

        distances, indices = collaborative_model.kneighbors(
            item_user_matrix[movie_index],
            n_neighbors=11
        )

        similarities = 1 - distances[0]

        for similar_index, similarity in zip(
            indices[0],
            similarities
        ):

            # Exclude the movie itself and movies already rated
            if similar_index == movie_index:
                continue

            if similar_index not in rated_movie_set:

                scores[similar_index] = (
                    scores.get(similar_index, 0)
                    + similarity * rating
                )

    if not scores:
        return pd.DataFrame()

    recommendations = pd.DataFrame(
        scores.items(),
        columns=["movie_index", "collaborative_score"]
    )

    recommendations = recommendations.sort_values(
        "collaborative_score",
        ascending=False
    ).head(n)

    recommendations["movieId"] = movie_encoder.inverse_transform(
        recommendations["movie_index"].to_numpy()
    )

    recommendations = recommendations.merge(
        movies,
        on="movieId",
        how="left"
    )

    return recommendations[
        ["movieId", "title", "genres", "collaborative_score"]
    ]
