
import os
import joblib
import pandas as pd
import streamlit as st

from src.content_based import get_similar_movies
from src.collaborative import get_collaborative_recommendations
from src.hybrid import get_hybrid_recommendations
from src.movie_cards import display_movie_cards

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #10101a 0%,
        #191927 55%,
        #10101a 100%
    );
    color: #f5f5f5;
}

/* Main content */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #151522;
    border-right: 1px solid #29293b;
}

/* Headings */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #e50914, #b20710);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #ff2633;
    color: white;
    border: none;
    transform: scale(1.02);
}

/* Input fields */
.stTextInput input,
.stNumberInput input {
    background-color: #222232;
    color: white;
    border: 1px solid #39394d;
    border-radius: 8px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #303044;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #343449;
}

/* Cinematic hero section */
.hero {
    padding: 35px 40px;
    margin-bottom: 30px;
    border-radius: 18px;
    background: linear-gradient(
        110deg,
        #301018,
        #171725 75%
    );
    border: 1px solid #49242e;
}

.hero-tag {
    color: #ff5964;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
    margin-bottom: 12px;
}

.hero h1 {
    font-size: 38px;
    margin: 0;
    color: white;
}

.hero p {
    color: #c5c5d2;
    font-size: 16px;
    margin-top: 12px;
    margin-bottom: 0;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD SAVED ARTIFACTS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")


@st.cache_resource
def load_artifacts():

    content = joblib.load(
        os.path.join(ARTIFACT_DIR, "content_artifacts.joblib")
    )

    collaborative = joblib.load(
        os.path.join(ARTIFACT_DIR, "collaborative_artifacts.joblib")
    )

    train_ratings = pd.read_parquet(
        os.path.join(ARTIFACT_DIR, "train_ratings.parquet")
    )

    return content, collaborative, train_ratings


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-tag">MOVIE DISCOVERY ENGINE</div>
    <h1>Find Your Next Favourite Movie</h1>
    <p>
        Intelligent recommendations powered by Content-Based Filtering,
        Collaborative Filtering and Hybrid AI.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

try:

    with st.spinner("Loading recommendation models..."):
        content, collaborative, train_ratings = load_artifacts()

except Exception as error:

    st.error(f"Error loading artifacts: {error}")
    st.stop()


# --------------------------------------------------
# EXTRACT MODEL OBJECTS
# --------------------------------------------------

movies = content["movies"]
movie_features = content["movie_features"]

tfidf_matrix = content["tfidf_matrix"]
content_model = content["content_model"]

user_encoder = collaborative["user_encoder"]
movie_encoder = collaborative["movie_encoder"]

user_item_matrix = collaborative["user_item_matrix"]
item_user_matrix = collaborative["item_user_matrix"]

collaborative_model = collaborative["collaborative_model"]


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.header("Recommendation Settings")

recommendation_type = st.sidebar.selectbox(
    "Choose Recommendation Method",
    [
        "Similar Movies",
        "Collaborative Filtering",
        "Hybrid Recommendations"
    ]
)

recommendation_count = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)


# --------------------------------------------------
# CONTENT-BASED RECOMMENDATIONS
# --------------------------------------------------

if recommendation_type == "Similar Movies":

    st.subheader("Find Movies Similar to Your Choice")

    search_movie = st.text_input(
        "Enter a movie name",
        placeholder="Example: Matrix"
    )

    if search_movie.strip():

        matches = movies[
            movies["title"].str.contains(
                search_movie,
                case=False,
                na=False,
                regex=False
            )
        ]

        if matches.empty:

            st.warning("No matching movies found.")

        else:

            selected_movie = st.selectbox(
                "Select the movie",
                matches["title"].tolist()
            )

            if st.button("Find Similar Movies"):

                with st.spinner("Finding similar movies..."):

                    recommendations = get_similar_movies(
                        selected_movie,
                        movie_features,
                        content_model,
                        tfidf_matrix,
                        n=recommendation_count
                    )

                if recommendations is None or recommendations.empty:

                    st.warning("No similar movies found.")

                else:

                    st.success("Recommendations generated!")

                    display_movie_cards(recommendations)


# --------------------------------------------------
# COLLABORATIVE FILTERING
# --------------------------------------------------

elif recommendation_type == "Collaborative Filtering":

    st.subheader("Personalized Movie Recommendations")

    user_id = st.number_input(
        "Enter MovieLens User ID",
        min_value=1,
        value=20759,
        step=1
    )

    if st.button("Get Recommendations"):

        with st.spinner("Generating recommendations..."):

            recommendations = get_collaborative_recommendations(
                user_id=int(user_id),
                ratings=train_ratings,
                movies=movies,
                user_encoder=user_encoder,
                movie_encoder=movie_encoder,
                user_item_matrix=user_item_matrix,
                item_user_matrix=item_user_matrix,
                collaborative_model=collaborative_model,
                n=recommendation_count
            )

        if recommendations is None or recommendations.empty:

            st.warning(
                "No recommendations available. "
                "Check whether this user exists in the training data."
            )

        else:

            st.success("Recommendations generated!")

            display_movie_cards(recommendations)


# --------------------------------------------------
# HYBRID RECOMMENDATIONS
# --------------------------------------------------

else:

    st.subheader("Hybrid Personalized Recommendations")

    user_id = st.number_input(
        "Enter MovieLens User ID",
        min_value=1,
        value=20759,
        step=1
    )

    if st.button("Get Hybrid Recommendations"):

        with st.spinner("Combining recommendation models..."):

            recommendations = get_hybrid_recommendations(
                user_id=int(user_id),
                ratings=train_ratings,
                movies=movies,
                movie_features=movie_features,
                content_model=content_model,
                tfidf_matrix=tfidf_matrix,
                user_encoder=user_encoder,
                movie_encoder=movie_encoder,
                user_item_matrix=user_item_matrix,
                item_user_matrix=item_user_matrix,
                collaborative_model=collaborative_model,
                n=recommendation_count
            )

        if recommendations is None or recommendations.empty:

            st.warning("No hybrid recommendations available.")

        else:

            st.success("Hybrid recommendations generated!")

            display_movie_cards(recommendations)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "MovieLens 32M | Content-Based: 20% | Collaborative: 80%"
)
