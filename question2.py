import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_movie_data(movies, tags):
    """Preprocesses movie data for content-based recommendation system"""
    movies['genres_list'] = movies['genres'].apply(lambda x: x.split('|'))
    
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.str.lower())).reset_index()
    
    movies_with_tags = pd.merge(movies, movie_tags, on='movieId', how='left')
    movies_with_tags['tag'] = movies_with_tags['tag'].fillna('')
    
    movies_with_tags['content'] = (
        movies_with_tags['title'] + ' ' + 
        movies_with_tags['genres'].apply(lambda x: x.replace('|', ' ')) + ' ' + 
        movies_with_tags['tag']
    )
    
    return movies_with_tags

def create_tfidf_matrix(movies_with_tags):
    """Creates TF-IDF matrix from movie descriptions"""
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=3,
        max_features=5000
    )
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_with_tags['content'])
    
    print(f"TF-IDF matrix dimensions: {tfidf_matrix.shape}")
    print(f"Number of features: {len(tfidf_vectorizer.get_feature_names_out())}")
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print("\nExamples of terms (features):")
    print(feature_names[:20])
    
    return tfidf_matrix, tfidf_vectorizer, feature_names

def get_movie_recommendations(movie_id, movies_with_tags, tfidf_matrix, n_recommendations=5):
    """Returns the n most similar movies to a given movie based on content"""
    movie_idx = movies_with_tags[movies_with_tags['movieId'] == movie_id].index
    
    if len(movie_idx) == 0:
        print(f"Movie with ID {movie_id} not found in the dataset.")
        return []
    
    movie_idx = movie_idx[0]
    
    movie_similarities = cosine_similarity(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()
    
    similar_indices = movie_similarities.argsort()[::-1][1:n_recommendations+1]
    
    recommended_movies = movies_with_tags.iloc[similar_indices][['movieId', 'title', 'genres']]
    
    recommended_movies['similarity_score'] = movie_similarities[similar_indices]
    
    return recommended_movies

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
    
    movies_with_tags = preprocess_movie_data(movies, tags)
    
    print("\nPreview of movies with preprocessed content:")
    print(movies_with_tags[['movieId', 'title', 'genres', 'content']].head())
    
    tfidf_matrix, tfidf_vectorizer, feature_names = create_tfidf_matrix(movies_with_tags)
    
    popular_movie_id = 1 # Toy Story
    movie_title = movies_with_tags[movies_with_tags['movieId'] == popular_movie_id]['title'].values[0]
    print(f"\nMovie selected for recommendations: {movie_title} (ID: {popular_movie_id})")
    
    recommendations = get_movie_recommendations(popular_movie_id, movies_with_tags, tfidf_matrix)
    
    print("\nContent-based recommended movies:")
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {row['title']} - {row['genres']} (Score: {row['similarity_score']:.4f})")
    
    explain_content_based_approach()
    
    return movies_with_tags, tfidf_matrix, tfidf_vectorizer, feature_names 