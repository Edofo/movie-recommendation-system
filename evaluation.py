import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
import time
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_rating_prediction(ratings_df, test_size=0.2, random_state=42, max_samples=None):
    """
    Evaluates rating prediction using RMSE
    
    Args:
        ratings_df: DataFrame with columns 'userId', 'movieId', 'rating'
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        max_samples: Maximum number of samples to use (None for all)
        
    Returns:
        Dictionary with RMSE value
    """
    start_time = time.time()
    print("\nEvaluating rating prediction (RMSE)...")
    
    if max_samples and len(ratings_df) > max_samples:
        ratings_df = ratings_df.sample(max_samples, random_state=random_state)
    
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    trainset, testset = surprise_train_test_split(data, test_size=test_size, random_state=random_state)
    
    algo = SVD()
    algo.fit(trainset)
    
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    
    elapsed_time = time.time() - start_time
    print(f"RMSE: {rmse:.4f} (calculated in {elapsed_time:.2f} seconds)")
    
    return {"rmse": rmse, "predictions": predictions}


def evaluate_recommendation_relevance(utility_matrix, ratings_df, threshold=3.5, test_size=0.2, random_state=42, max_users=100):
    """
    Evaluates recommendation relevance using Precision, Recall and F1-score
    
    Args:
        utility_matrix: The user-item matrix
        ratings_df: DataFrame with columns 'userId', 'movieId', 'rating'
        threshold: Rating threshold to consider an item relevant
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        max_users: Maximum number of users to evaluate (None for all)
        
    Returns:
        Dictionary with precision, recall and f1-score values
    """
    start_time = time.time()
    print("\nEvaluating recommendation relevance (Precision, Recall, F1)...")
    
    train_df, test_df = train_test_split(
        ratings_df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=ratings_df['userId']
    )
    
    train_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating')
    
    test_users = test_df['userId'].unique()
    if max_users and len(test_users) > max_users:
        test_users = np.random.choice(test_users, max_users, replace=False)
    
    all_precision = []
    all_recall = []
    all_f1 = []
    
    k = 10
    
    print(f"Evaluating {len(test_users)} users...")
    
    for user_idx, user_id in enumerate(test_users):
        if user_idx % 10 == 0:
            print(f"Processing user {user_idx}/{len(test_users)}...")
            
        if user_id not in train_matrix.index:
            continue
        
        user_vec = train_matrix.loc[user_id].fillna(0).values
        similarities = {}
        
        other_users = [u for u in train_matrix.index if u != user_id]
        if len(other_users) > 100:
            other_users = np.random.choice(other_users, 100, replace=False)
            
        for other_user in other_users:
            other_vec = train_matrix.loc[other_user].fillna(0).values
            
            common_mask = (~np.isnan(train_matrix.loc[user_id])) & (~np.isnan(train_matrix.loc[other_user]))
            if common_mask.sum() < 5:
                continue
                
            user_common = train_matrix.loc[user_id][common_mask].values
            other_common = train_matrix.loc[other_user][common_mask].values
            
            dot_product = np.dot(user_common, other_common)
            user_norm = np.linalg.norm(user_common)
            other_norm = np.linalg.norm(other_common)
            
            if user_norm == 0 or other_norm == 0:
                continue
                
            similarity = dot_product / (user_norm * other_norm)
            
            if similarity > 0:
                similarities[other_user] = similarity
        
        recommendations = {}
        
        for other_user, sim in similarities.items():
            user_null_mask = np.isnan(train_matrix.loc[user_id])
            other_rated_mask = ~np.isnan(train_matrix.loc[other_user])
            
            candidate_movie_indices = np.where(user_null_mask & other_rated_mask)[0]
            
            for idx in candidate_movie_indices:
                movie_id = train_matrix.columns[idx]
                if movie_id not in recommendations:
                    recommendations[movie_id] = 0
                recommendations[movie_id] += sim * train_matrix.loc[other_user, movie_id]
        
        if not recommendations:
            continue
            
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:k]
        recommended_movie_ids = [movie_id for movie_id, _ in sorted_recommendations]
        
        relevant_movies = test_df[(test_df['userId'] == user_id) & (test_df['rating'] >= threshold)]['movieId'].tolist()
        
        if not relevant_movies:
            continue
            
        all_test_movies = test_df[test_df['userId'] == user_id]['movieId'].unique()
        y_true = np.zeros(len(all_test_movies))
        y_pred = np.zeros(len(all_test_movies))
        
        for movie_idx, movie_id in enumerate(all_test_movies):
            if movie_id in relevant_movies:
                y_true[movie_idx] = 1
            if movie_id in recommended_movie_ids:
                y_pred[movie_idx] = 1
        
        if np.sum(y_pred) > 0:
            all_precision.append(precision_score(y_true, y_pred, zero_division=0))
            all_recall.append(recall_score(y_true, y_pred, zero_division=0))
            all_f1.append(f1_score(y_true, y_pred, zero_division=0))
    
    avg_precision = np.mean(all_precision) if all_precision else 0
    avg_recall = np.mean(all_recall) if all_recall else 0
    avg_f1 = np.mean(all_f1) if all_f1 else 0
    
    elapsed_time = time.time() - start_time
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"F1-score@{k}: {avg_f1:.4f}")
    print(f"Calculated in {elapsed_time:.2f} seconds")
    
    return {
        "precision": avg_precision, 
        "recall": avg_recall, 
        "f1": avg_f1
    }


def calculate_ap(y_true, y_score, k=10):
    """
    Calculate Average Precision at k
    
    Args:
        y_true: Binary relevance vector
        y_score: Predicted scores
        k: Number of items to consider
    
    Returns:
        Average Precision at k
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    
    precisions = [np.mean(y_true[:i+1]) for i in range(len(y_true)) if y_true[i]]
    
    return np.mean(precisions) if precisions else 0


def calculate_dcg(y_true, y_score, k=10):
    """
    Calculate Discounted Cumulative Gain
    
    Args:
        y_true: Binary relevance vector
        y_score: Predicted scores
        k: Number of items to consider
    
    Returns:
        DCG@k
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    
    gains = y_true
    discounts = np.log2(np.arange(2, len(y_true) + 2))
    return np.sum(gains / discounts)


def calculate_ndcg(y_true, y_score, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain
    
    Args:
        y_true: Binary relevance vector
        y_score: Predicted scores
        k: Number of items to consider
    
    Returns:
        NDCG@k
    """
    dcg = calculate_dcg(y_true, y_score, k)
    
    ideal_order = np.argsort(y_true)[::-1]
    ideal_y_true = np.take(y_true, ideal_order[:k])
    idcg = calculate_dcg(ideal_y_true, ideal_y_true, k)
    
    return dcg / idcg if idcg > 0 else 0


def evaluate_recommendation_ranking(utility_matrix, ratings_df, movies_with_tags, tfidf_matrix, threshold=3.5, test_size=0.2, random_state=42, max_users=100, max_candidates=1000, max_liked=5):
    """
    Evaluates recommendation ranking using MAP and NDCG
    
    Args:
        utility_matrix: The user-item matrix
        ratings_df: DataFrame with columns 'userId', 'movieId', 'rating'
        movies_with_tags: DataFrame with processed movie data
        tfidf_matrix: TF-IDF matrix for content-based recommendations
        threshold: Rating threshold to consider an item relevant
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        max_users: Maximum number of users to evaluate
        max_candidates: Maximum number of candidate movies to evaluate
        max_liked: Maximum number of liked movies to use as reference
        
    Returns:
        Dictionary with MAP and NDCG values
    """
    start_time = time.time()
    print("\nEvaluating recommendation ranking (MAP, NDCG)...")
    
    train_df, test_df = train_test_split(
        ratings_df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=ratings_df['userId']
    )
    
    train_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating')
    
    test_users = test_df['userId'].unique()
    if max_users and len(test_users) > max_users:
        test_users = np.random.choice(test_users, max_users, replace=False)
    
    all_map = []
    all_ndcg = []
    
    k = 10
    
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies_with_tags['movieId'])}
    
    print(f"Evaluating {len(test_users)} users...")
    
    for user_idx, user_id in enumerate(test_users):
        if user_idx % 10 == 0:
            print(f"Processing user {user_idx}/{len(test_users)}...")
            
        if user_id not in train_matrix.index:
            continue
            
        user_rated = train_df[train_df['userId'] == user_id]['movieId'].unique()
        
        user_test = test_df[test_df['userId'] == user_id]
        
        if len(user_test) == 0:
            continue
            
        user_liked = train_df[(train_df['userId'] == user_id) & (train_df['rating'] >= threshold)]['movieId'].values
        
        if len(user_liked) == 0:
            continue
            
        if len(user_liked) > max_liked:
            user_liked = np.random.choice(user_liked, max_liked, replace=False)
            
        liked_indices = []
        for movie_id in user_liked:
            if movie_id in movie_to_idx:
                idx = movie_to_idx[movie_id]
                if idx < tfidf_matrix.shape[0]:
                    liked_indices.append(idx)
                    
        if not liked_indices:
            continue
            
        candidate_movies = []
        for movie_id in train_matrix.columns:
            if (movie_id not in user_rated and 
                movie_id in movie_to_idx and 
                movie_to_idx[movie_id] < tfidf_matrix.shape[0]):
                candidate_movies.append(movie_id)
                
        if len(candidate_movies) > max_candidates:
            candidate_movies = np.random.choice(candidate_movies, max_candidates, replace=False)
        
        recommendations = {}
        for movie_id in candidate_movies:
            movie_idx = movie_to_idx[movie_id]
            
            total_sim = 0
            valid_count = 0
            
            for liked_idx in liked_indices:
                sim = cosine_similarity(tfidf_matrix[liked_idx:liked_idx+1], 
                                       tfidf_matrix[movie_idx:movie_idx+1])[0][0]
                if sim > 0:
                    total_sim += sim
                    valid_count += 1
                    
            if valid_count > 0:
                recommendations[movie_id] = total_sim / valid_count
        
        if not recommendations:
            continue
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:k]
        
        test_movies = user_test['movieId'].unique()
        y_true = np.zeros(len(test_movies))
        y_score = np.zeros(len(test_movies))
        
        for movie_idx, movie_id in enumerate(test_movies):
            rating = user_test[user_test['movieId'] == movie_id]['rating'].values[0]
            y_true[movie_idx] = 1 if rating >= threshold else 0
            
            y_score[movie_idx] = recommendations.get(movie_id, 0)
        
        if np.sum(y_true) > 0 and np.sum(y_score) > 0:
            ap = calculate_ap(y_true, y_score, k)
            ndcg = calculate_ndcg(y_true, y_score, k)
            
            all_map.append(ap)
            all_ndcg.append(ndcg)
    
    map_score = np.mean(all_map) if all_map else 0
    ndcg_score = np.mean(all_ndcg) if all_ndcg else 0
    
    elapsed_time = time.time() - start_time
    print(f"MAP@{k}: {map_score:.4f}")
    print(f"NDCG@{k}: {ndcg_score:.4f}")
    print(f"Calculated in {elapsed_time:.2f} seconds")
    
    return {
        "map": map_score, 
        "ndcg": ndcg_score
    }


def run_evaluation(ratings, movies, utility_matrix, movies_with_tags, tfidf_matrix, max_users=50, max_samples=10000):
    """
    Runs all evaluation metrics
    
    Args:
        ratings: DataFrame with ratings data
        movies: DataFrame with movies data
        utility_matrix: User-item utility matrix
        movies_with_tags: DataFrame with processed movie data including tags
        tfidf_matrix: TF-IDF matrix for content-based recommendations
        max_users: Maximum number of users to evaluate for relevance and ranking metrics
        max_samples: Maximum number of samples to use for RMSE evaluation        
    Returns:
        Dictionary with all evaluation results
    """

    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)

    print(f"\nRunning evaluation with max_users={max_users}, max_samples={max_samples}")
    
    rmse_results = evaluate_rating_prediction(ratings, max_samples=max_samples)
    
    relevance_results = evaluate_recommendation_relevance(utility_matrix, ratings, max_users=max_users)
    
    max_candidates = 500  
    max_liked = 3         
    ranking_results = evaluate_recommendation_ranking(
        utility_matrix, 
        ratings, 
        movies_with_tags, 
        tfidf_matrix, 
        max_users=max_users,
        max_candidates=max_candidates,
        max_liked=max_liked
    )
    
    all_results = {**rmse_results, **relevance_results, **ranking_results}
    
    print("\n" + "-"*50)
    print("EVALUATION SUMMARY")
    print("-"*50)
    print(f"RMSE: {all_results['rmse']:.4f}")
    print(f"Precision: {all_results['precision']:.4f}")
    print(f"Recall: {all_results['recall']:.4f}")
    print(f"F1-score: {all_results['f1']:.4f}")
    print(f"MAP: {all_results['map']:.4f}")
    print(f"NDCG: {all_results['ndcg']:.4f}")
    
    return all_results