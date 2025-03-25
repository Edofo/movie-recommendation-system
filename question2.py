#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_movie_data(movies, tags):
    """Preprocesses movie data for content-based recommendation system"""
    # Create a 'genres_list' column containing the list of genres
    movies['genres_list'] = movies['genres'].apply(lambda x: x.split('|'))
    
    # Aggregate tags by movie
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.str.lower())).reset_index()
    
    # Merge movies and tags data
    movies_with_tags = pd.merge(movies, movie_tags, on='movieId', how='left')
    movies_with_tags['tag'] = movies_with_tags['tag'].fillna('')
    
    # Create a 'content' column combining title, genres and tags
    movies_with_tags['content'] = (
        movies_with_tags['title'] + ' ' + 
        movies_with_tags['genres'].apply(lambda x: x.replace('|', ' ')) + ' ' + 
        movies_with_tags['tag']
    )
    
    return movies_with_tags

def create_tfidf_matrix(movies_with_tags):
    """Creates TF-IDF matrix from movie descriptions"""
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=3,  # Ignore terms appearing in fewer than 3 documents
        max_features=5000  # Limit to 5000 features to avoid a too large matrix
    )
    
    # Create TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_with_tags['content'])
    
    print(f"TF-IDF matrix dimensions: {tfidf_matrix.shape}")
    print(f"Number of features: {len(tfidf_vectorizer.get_feature_names_out())}")
    
    # Display some examples of terms (features)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print("\nExamples of terms (features):")
    print(feature_names[:20])
    
    return tfidf_matrix, tfidf_vectorizer, feature_names

def get_movie_recommendations(movie_id, movies_with_tags, tfidf_matrix, n_recommendations=5):
    """Returns the n most similar movies to a given movie based on content"""
    # Find the index of the movie in the DataFrame
    movie_idx = movies_with_tags[movies_with_tags['movieId'] == movie_id].index
    
    if len(movie_idx) == 0:
        print(f"Movie with ID {movie_id} not found in the dataset.")
        return []
    
    movie_idx = movie_idx[0]
    
    # Calculate cosine similarity between this movie and all others
    movie_similarities = cosine_similarity(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()
    
    # Get indices of most similar movies (excluding the movie itself)
    similar_indices = movie_similarities.argsort()[::-1][1:n_recommendations+1]
    
    # Retrieve information about recommended movies
    recommended_movies = movies_with_tags.iloc[similar_indices][['movieId', 'title', 'genres']]
    
    # Add similarity score
    recommended_movies['similarity_score'] = movie_similarities[similar_indices]
    
    return recommended_movies

def visualize_important_features(movie_id, movies_with_tags, tfidf_matrix, tfidf_vectorizer, feature_names, top_n=10):
    """Visualizes the most important features for a given movie"""
    # Find the index of the movie in the DataFrame
    movie_idx = movies_with_tags[movies_with_tags['movieId'] == movie_id].index
    
    if len(movie_idx) == 0:
        print(f"Movie with ID {movie_id} not found in the dataset.")
        return
    
    movie_idx = movie_idx[0]
    movie_title = movies_with_tags.iloc[movie_idx]['title']
    
    # Get TF-IDF vector for this movie
    movie_vector = tfidf_matrix[movie_idx].toarray().flatten()
    
    # Get indices of most important features
    top_feature_indices = movie_vector.argsort()[::-1][:top_n]
    
    # Get names and scores of most important features
    top_features = [(feature_names[idx], movie_vector[idx]) for idx in top_feature_indices]
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.barh([feature for feature, _ in top_features], [score for _, score in top_features])
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Term')
    plt.title(f'Top {top_n} most important features for\n"{movie_title}"')
    plt.tight_layout()
    plt.show()

def explain_content_based_approach():
    """Explains advantages and limitations of content-based approach"""
    print("\n" + "="*50)
    print("ADVANTAGES AND LIMITATIONS OF CONTENT-BASED APPROACH")
    print("="*50)
    
    print("\nAdvantages:")
    advantages = [
        "1. No need for other user data (solves cold start for new users)",
        "2. Recommendation transparency: can explain why an item is recommended",
        "3. Ability to recommend niche or new items (solves cold start for new items)",
        "4. Independence from majority preferences (less popularity bias)",
        "5. Privacy-friendly: doesn't require data about other users"
    ]
    for adv in advantages:
        print(adv)
    
    print("\nLimitations:")
    limitations = [
        "1. Overspecialization / lack of diversity: tends to recommend similar items",
        "2. Difficulty capturing complex qualitative aspects (quality, style, emotions...)",
        "3. Dependence on metadata quality and representation",
        "4. Inability to detect preferences not explicitly represented in features",
        "5. Doesn't benefit from collective wisdom or trends"
    ]
    for lim in limitations:
        print(lim)

def run_question2(movies, tags):
    """Executes Question 2"""
    print("\n" + "="*50)
    print("QUESTION 2: CONTENT-BASED RECOMMENDATION")
    print("="*50)
    
    # Preprocess data for content-based approach
    movies_with_tags = preprocess_movie_data(movies, tags)
    
    print("\nPreview of movies with preprocessed content:")
    print(movies_with_tags[['movieId', 'title', 'genres', 'content']].head())
    
    # Create TF-IDF matrix
    tfidf_matrix, tfidf_vectorizer, feature_names = create_tfidf_matrix(movies_with_tags)
    
    # Choose a popular movie for demonstration
    popular_movie_id = 1 # Toy Story
    movie_title = movies_with_tags[movies_with_tags['movieId'] == popular_movie_id]['title'].values[0]
    print(f"\nMovie selected for recommendations: {movie_title} (ID: {popular_movie_id})")
    
    # Get recommendations for this movie
    recommendations = get_movie_recommendations(popular_movie_id, movies_with_tags, tfidf_matrix)
    
    print("\nContent-based recommended movies:")
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {row['title']} - {row['genres']} (Score: {row['similarity_score']:.4f})")
    
    # Visualize important features for this movie
    try:
        visualize_important_features(popular_movie_id, movies_with_tags, tfidf_matrix, tfidf_vectorizer, feature_names)
    except Exception as e:
        print(f"Error during visualization: {e}")
    
    # Explain advantages and limitations of content-based approach
    explain_content_based_approach()
    
    return movies_with_tags, tfidf_matrix, tfidf_vectorizer, feature_names 