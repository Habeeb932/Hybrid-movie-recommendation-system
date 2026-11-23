
import os
import pandas as pd


def load_movielens_data(data_folder):
    """Load MovieLens datasets."""

    movies = pd.read_csv(os.path.join(data_folder, "movies.csv"))
    ratings = pd.read_csv(os.path.join(data_folder, "ratings.csv"))
    tags = pd.read_csv(os.path.join(data_folder, "tags.csv"))
    links = pd.read_csv(os.path.join(data_folder, "links.csv"))

    return movies, ratings, tags, links


def clean_movie_tags(tags):
    """Clean movie tags."""

    clean_tags = tags.dropna(subset=["tag"]).copy()

    clean_tags["tag"] = (
        clean_tags["tag"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    clean_tags = clean_tags[clean_tags["tag"] != ""]

    clean_tags = clean_tags.drop_duplicates(
        subset=["movieId", "tag"]
    )

    return clean_tags


def create_movie_features(movies, clean_tags):
    """Combine movie genres and tags into text features."""

    movie_tag_features = (
        clean_tags.groupby("movieId")["tag"]
        .apply(lambda movie_tags: " ".join(movie_tags))
        .reset_index(name="tag_text")
    )

    movie_features = movies[
        ["movieId", "title", "genres"]
    ].copy()

    movie_features["genres"] = (
        movie_features["genres"].fillna("")
    )

    movie_features["genre_text"] = (
        movie_features["genres"]
        .str.replace("|", " ", regex=False)
    )

    movie_features = movie_features.merge(
        movie_tag_features,
        on="movieId",
        how="left"
    )

    movie_features["tag_text"] = (
        movie_features["tag_text"].fillna("")
    )

    movie_features["combined_features"] = (
        movie_features["genre_text"]
        + " "
        + movie_features["tag_text"]
    ).str.strip()

    return movie_features
