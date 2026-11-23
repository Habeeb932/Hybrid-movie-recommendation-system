
import pandas as pd

from src.content_based import get_similar_movies
from src.collaborative import get_collaborative_recommendations


def get_hybrid_recommendations(
    user_id,
    ratings,
    movies,
    movie_features,
    content_model,
    tfidf_matrix,
    user_encoder,
    movie_encoder,
    user_item_matrix,
    item_user_matrix,
    collaborative_model,
    n=10,
    content_weight=0.2,
    collaborative_weight=0.8,
    rrf_constant=60
):

    # Step 1: Collaborative recommendations
    collaborative_results = get_collaborative_recommendations(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        user_encoder=user_encoder,
        movie_encoder=movie_encoder,
        user_item_matrix=user_item_matrix,
        item_user_matrix=item_user_matrix,
        collaborative_model=collaborative_model,
        n=50
    )

    if collaborative_results is None:
        collaborative_results = pd.DataFrame()

    # Step 2: Identify movies already rated
    user_rated = set(
        ratings.loc[
            ratings["userId"] == user_id,
            "movieId"
        ]
    )

    # Step 3: Content-based recommendations
    content_scores = {}

    user_movies = ratings[
        (ratings["userId"] == user_id)
        & (ratings["rating"] >= 4.0)
    ]

    for movie_id in user_movies["movieId"].values:

        movie_match = movie_features[
            movie_features["movieId"] == movie_id
        ]

        if movie_match.empty:
            continue

        movie_title = movie_match.iloc[0]["title"]

        similar_movies = get_similar_movies(
            movie_title,
            movie_features,
            content_model,
            tfidf_matrix,
            n=50
        )

        if similar_movies is None:
            continue

        for rank, row in enumerate(
            similar_movies.itertuples(index=False),
            start=1
        ):

            recommended_id = row.movieId

            if recommended_id not in user_rated:

                content_scores[recommended_id] = min(
                    content_scores.get(
                        recommended_id,
                        float("inf")
                    ),
                    rank
                )

    # Step 4: Combine using weighted Reciprocal Rank Fusion
    hybrid_scores = {}

    for rank, row in enumerate(
        collaborative_results.itertuples(index=False),
        start=1
    ):

        movie_id = row.movieId

        hybrid_scores[movie_id] = (
            hybrid_scores.get(movie_id, 0)
            + collaborative_weight / (rrf_constant + rank)
        )

    for movie_id, rank in content_scores.items():

        hybrid_scores[movie_id] = (
            hybrid_scores.get(movie_id, 0)
            + content_weight / (rrf_constant + rank)
        )

    if not hybrid_scores:
        return pd.DataFrame(
            columns=["movieId", "title", "genres", "hybrid_score"]
        )

    # Step 5: Create final recommendations
    recommendations = pd.DataFrame(
        hybrid_scores.items(),
        columns=["movieId", "hybrid_score"]
    )

    recommendations = recommendations.sort_values(
        "hybrid_score",
        ascending=False
    )

    recommendations = recommendations.merge(
        movies,
        on="movieId",
        how="left"
    )

    recommendations = recommendations[
        ~recommendations["movieId"].isin(user_rated)
    ]

    return recommendations[
        ["movieId", "title", "genres", "hybrid_score"]
    ].head(n)
