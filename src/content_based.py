
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def build_content_model(movie_features):

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        max_features=100000
    )

    tfidf_matrix = tfidf.fit_transform(
        movie_features["combined_features"]
    )

    content_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=11,
        n_jobs=-1
    )

    content_model.fit(tfidf_matrix)

    return tfidf, tfidf_matrix, content_model


def get_similar_movies(
    movie_title,
    movie_features,
    content_model,
    tfidf_matrix,
    n=10
):

    matches = movie_features[
        movie_features["title"].str.lower() == movie_title.lower()
    ]

    if matches.empty:
        return None

    movie_index = matches.index[0]

    total_movies = tfidf_matrix.shape[0]
    neighbors_to_find = min(n + 1, total_movies)

    distances, indices = content_model.kneighbors(
        tfidf_matrix[movie_index],
        n_neighbors=neighbors_to_find
    )

    # Exclude the selected movie itself
    results = [
        (index, distance)
        for index, distance in zip(indices[0], distances[0])
        if index != movie_index
    ][:n]

    if not results:
        return None

    recommended_indices = [index for index, _ in results]
    similarities = [1 - distance for _, distance in results]

    recommendations = movie_features.iloc[
        recommended_indices
    ].copy()

    recommendations["similarity_score"] = similarities

    return recommendations[
        ["movieId", "title", "genres", "similarity_score"]
    ]
