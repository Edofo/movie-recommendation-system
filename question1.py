import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

def create_utility_matrix(ratings):
    """Creates a utility matrix (users as rows, movies as columns)"""
    utility_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
    return utility_matrix

def calculate_user_similarity(utility_matrix, user1_id, user2_id):
    """Calculates similarity between two users using Cosine and Pearson metrics"""
    user1_ratings = utility_matrix.loc[user1_id].values
    user2_ratings = utility_matrix.loc[user2_id].values
    
    common_movies_mask = ~np.isnan(user1_ratings) & ~np.isnan(user2_ratings)
    user1_common = user1_ratings[common_movies_mask]
    user2_common = user2_ratings[common_movies_mask]
    
    n_common = np.sum(common_movies_mask)
    if n_common == 0:
        return 0, 0, 0  # No movies in common
    
    cosine_sim = 1 - cosine(user1_common, user2_common)
    pearson_sim, _ = pearsonr(user1_common, user2_common)
    
    return cosine_sim, pearson_sim, n_common

def calculate_movie_similarity(utility_matrix, movies_df, movie1_id, movie2_id):
    """Calculates similarity between two movies using Cosine and Pearson metrics"""
    if movie1_id not in utility_matrix.columns or movie2_id not in utility_matrix.columns:
        print(f"One of the movies (ID: {movie1_id} or {movie2_id}) is not in the utility matrix")
        return 0, 0, 0, "", ""
    
    movie1_ratings = utility_matrix[movie1_id].values
    movie2_ratings = utility_matrix[movie2_id].values
    
    common_users_mask = ~np.isnan(movie1_ratings) & ~np.isnan(movie2_ratings)
    movie1_common = movie1_ratings[common_users_mask]
    movie2_common = movie2_ratings[common_users_mask]
    
    n_common = np.sum(common_users_mask)
    if n_common == 0:
        return 0, 0, 0, "", ""  # No users in common
    
    movie1_title = movies_df[movies_df['movieId'] == movie1_id]['title'].values[0]
    movie2_title = movies_df[movies_df['movieId'] == movie2_id]['title'].values[0]
    
    cosine_sim = 1 - cosine(movie1_common, movie2_common)
    pearson_sim, _ = pearsonr(movie1_common, movie2_common)
    
    return cosine_sim, pearson_sim, n_common, movie1_title, movie2_title

def get_interpretation(score, thresholds, messages):
    """Returns interpretation based on score and defined thresholds"""
    for i, threshold in enumerate(thresholds):
        if score > threshold:
            return messages[i]
    return messages[-1]  # Default case with -inf

def interpret_similarity_results(cosine_sim_users, pearson_sim_users, cosine_sim_movies, pearson_sim_movies):
    """Interprets similarity results"""
    cosine_thresholds = [0.7, 0.4, -float('inf')]
    cosine_messages = {
        'user': [
            "Strong similarity in preferences",
            "Moderate similarity in preferences", 
            "Weak similarity in preferences"
        ],
        'movie': [
            "Very similar movies in their evaluations", 
            "Moderately similar movies", 
            "Dissimilar movies"
        ]
    }

    pearson_thresholds = [0.7, 0.3, -0.3, -0.7, -float('inf')]
    pearson_messages = {
        'user': [
            "Strong positive correlation (very similar tastes)",
            "Moderate positive correlation",
            "Weak or no correlation",
            "Moderate negative correlation (opposite tastes)",
            "Strong negative correlation (very opposite tastes)"
        ],
        'movie': [
            "Strongly correlated movies (appreciated by same groups)",
            "Moderately correlated movies",
            "Weak or no correlation between movies",
            "Negatively correlated movies (appreciated by different groups)",
            "Strongly negatively correlated movies"
        ]
    }

    print("\n--- Result Interpretation ---")
    print("User similarity:")
    cosine_msg = get_interpretation(cosine_sim_users, cosine_thresholds, cosine_messages['user'])
    print(f"- Cosine: {cosine_sim_users:.4f} - {cosine_msg}")

    pearson_msg = get_interpretation(pearson_sim_users, pearson_thresholds, pearson_messages['user'])
    print(f"- Pearson: {pearson_sim_users:.4f} - {pearson_msg}")

    print("\nMovie similarity:")
    cosine_msg = get_interpretation(cosine_sim_movies, cosine_thresholds, cosine_messages['movie'])
    print(f"- Cosine: {cosine_sim_movies:.4f} - {cosine_msg}")

    pearson_msg = get_interpretation(pearson_sim_movies, pearson_thresholds, pearson_messages['movie'])
    print(f"- Pearson: {pearson_sim_movies:.4f} - {pearson_msg}")

def run_question1(ratings, movies):
    """Executes Question 1"""
    print("\n" + "="*50)
    print("QUESTION 1: UTILITY MATRIX AND SIMILARITY")
    print("="*50)
    
    print("\nRatings preview:")
    print(ratings.head())
    
    print("\nMovies preview:")
    print(movies.head())
    
    utility_matrix = create_utility_matrix(ratings)
    
    print("\nUtility matrix dimensions:", utility_matrix.shape)
    print("\nUtility matrix preview:")
    print(utility_matrix.iloc[:5, :5])  # Display first 5 rows and columns
    
    non_nan_values = utility_matrix.count().sum()
    total_cells = utility_matrix.size
    density = non_nan_values / total_cells * 100
    print(f"\nMatrix density: {density:.2f}% ({non_nan_values} ratings out of {total_cells} possible)")
    
    user1_id = 1
    user2_id = 2
    print(f"\nComparison between users {user1_id} and {user2_id}")
    
    cosine_sim_users, pearson_sim_users, n_common_movies = calculate_user_similarity(utility_matrix, user1_id, user2_id)
    print(f"Number of movies rated in common: {n_common_movies}")
    print(f"Cosine similarity between users: {cosine_sim_users:.4f}")
    print(f"Pearson correlation between users: {pearson_sim_users:.4f}")
    
    popular_movies = ratings.groupby('movieId').count().sort_values('rating', ascending=False).head(10)
    movie1_id = popular_movies.index[0]
    movie2_id = popular_movies.index[1]
    
    cosine_sim_movies, pearson_sim_movies, n_common_users, movie1_title, movie2_title = calculate_movie_similarity(
        utility_matrix, movies, movie1_id, movie2_id
    )
    
    print(f"\nComparison between movies: {movie1_title} (ID: {movie1_id}) and {movie2_title} (ID: {movie2_id})")
    print(f"Number of users who rated both movies: {n_common_users}")
    print(f"Cosine similarity between movies: {cosine_sim_movies:.4f}")
    print(f"Pearson correlation between movies: {pearson_sim_movies:.4f}")
    
    interpret_similarity_results(cosine_sim_users, pearson_sim_users, cosine_sim_movies, pearson_sim_movies)
    
    return utility_matrix 