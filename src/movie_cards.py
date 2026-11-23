
import streamlit as st


def display_movie_cards(recommendations, columns_per_row=3):

    if recommendations is None or recommendations.empty:
        st.info("No movies to display.")
        return

    movies = recommendations.to_dict("records")

    for start in range(0, len(movies), columns_per_row):

        cols = st.columns(columns_per_row)

        for col, movie in zip(
            cols,
            movies[start:start + columns_per_row]
        ):

            with col:

                with st.container(border=True):

                    title = movie.get("title", "Unknown Movie")
                    genres = movie.get("genres", "Genres unavailable")

                    # Movie title
                    st.markdown(f"### {title}")

                    # Genres
                    st.caption(str(genres).replace("|", " • "))

                    # Recommendation score
                    if "similarity_score" in movie:
                        st.metric(
                            "Similarity",
                            f"{movie['similarity_score']:.3f}"
                        )

                    elif "hybrid_score" in movie:
                        st.metric(
                            "Hybrid Score",
                            f"{movie['hybrid_score']:.4f}"
                        )

                    elif "collaborative_score" in movie:
                        st.metric(
                            "Collaborative Score",
                            f"{movie['collaborative_score']:.3f}"
                        )

