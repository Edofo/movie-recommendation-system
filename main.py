import argparse
import sys
from data_loader import load_movielens_data
from question1 import run_question1
from question2 import run_question2
from evaluation import run_evaluation

def main():
    """Main function executing different parts of the project"""
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation metrics")
    parser.add_argument("--max-users", type=int, default=50, help="Maximum number of users for evaluation")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum number of samples for RMSE evaluation")
    args = parser.parse_args()
    
    print("Collaborative Filtering and Content-Based Recommendation")
    
    try:
        ratings, movies, tags, path = load_movielens_data()
        print(f"\nData loaded from: {path}")
        
        utility_matrix = run_question1(ratings, movies)
        
        movies_with_tags, tfidf_matrix, _tfidf_vectorizer, _feature_names = run_question2(movies, tags)
        
        if args.evaluate:
            run_evaluation(
                ratings, 
                movies, 
                utility_matrix, 
                movies_with_tags, 
                tfidf_matrix,
                max_users=args.max_users,
                max_samples=args.max_samples
            )
        
        print("\n" + "="*50)
        print("END OF PROJECT")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("If this is related to the evaluation metrics, try running with lower values for --max-users and --max-samples.")
        sys.exit(1)

if __name__ == "__main__":
    main()
