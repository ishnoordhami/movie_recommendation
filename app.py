import streamlit as st
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Page Configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    color: #FFD700;
    font-size: 50px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: white;
    font-size: 18px;
}

.stButton>button {
    background-color: #FFD700;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.recommend-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1F2937;
    margin-bottom: 10px;
    color: white;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# Load Files
movie_matrix = movie_matrix = pd.read_csv(
    "movie_matrix.csv",
    index_col=0
)
rating = pd.read_csv("ratings.csv")

df = pd.merge(ratings, movies, on="movieId")

movie_matrix = df.pivot_table(
    index="title",
    columns="userId",
    values="rating"
).fillna(0)

knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=6
)

knn.fit(movie_matrix)

# Title
st.markdown('<p class="title">🎬 Movie Recommendation System</p>',
            unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Recommendations Based on Movie Ratings Using KNN</p>',
    unsafe_allow_html=True
)

st.write("")

# Movie Dropdown
selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movie_matrix.index
)

# Recommendation Function
def recommend(movie_name):

    movie_index = movie_matrix.index.get_loc(movie_name)

    distances, indices = knn.kneighbors(
        movie_matrix.iloc[movie_index].values.reshape(1, -1),
        n_neighbors=6
    )

    recommendations = []

    for i in range(1, len(indices.flatten())):

        movie = movie_matrix.index[
            indices.flatten()[i]
        ]

        similarity = round(
            (1 - distances.flatten()[i]) * 100,
            2
        )

        recommendations.append(
            (movie, similarity)
        )

    return recommendations

# Button
if st.button("🎬 Recommend Movies"):

    recommendations = recommend(selected_movie)

    st.subheader("🍿 Recommended Movies")

    for movie, similarity in recommendations:

        st.markdown(
            f"""
            <div class="recommend-box">
            <b>{movie}</b><br>
            Similarity Score: {similarity}%
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")
st.caption("Built using K-Nearest Neighbors (KNN) and MovieLens Dataset")
