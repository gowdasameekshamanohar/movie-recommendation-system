import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("combined.csv")
    return data

data = load_data()

# -----------------------------
# USER-ITEM MATRIX
# -----------------------------
user_item_matrix = data.pivot_table(
    index="userId",
    columns="title",
    values="rating"
).fillna(0)

# -----------------------------
# COSINE SIMILARITY (MOVIE-MOVIE)
# -----------------------------
movie_similarity = cosine_similarity(user_item_matrix.T)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def recommend_movies(user_id, n=5):
    if user_id not in user_item_matrix.index:
        return None, "New user detected (Cold Start). Showing popular movies."

    user_ratings = user_item_matrix.loc[user_id]
    liked_movies = user_ratings[user_ratings > 0].index

    scores = pd.Series(dtype=float)

    for movie in liked_movies:
        scores = scores.add(movie_similarity_df[movie], fill_value=0)

    scores = scores.drop(liked_movies, errors="ignore")
    recommendations = scores.sort_values(ascending=False).head(n)

    return recommendations.index.tolist(), None

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Collaborative Filtering using Cosine Similarity")

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    step=1
)

if st.button("Get Recommendations"):
    movies, message = recommend_movies(user_id)

    if message:
        st.warning(message)

        popular_movies = (
            data.groupby("title")["rating"]
            .count()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )

        st.subheader("🔥 Popular Movies")
        for movie in popular_movies:
            st.write("•", movie)

    else:
        st.subheader("✅ Recommended Movies")
        for movie in movies:
            st.write("•", movie)