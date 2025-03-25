#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collaborative Filtering and Content-Based Recommendation
--------------------------------------------------------

This program explores recommendation techniques using the MovieLens dataset.
It implements a utility matrix and calculates different similarity measures,
then creates a content-based recommendation system using TF-IDF.

Author: [Your name]
Date: [Date]
"""

# Import custom modules
from data_loader import load_movielens_data
from question1 import run_question1
from question2 import run_question2

def main():
    """Main function executing different parts of the project"""
    print("Collaborative Filtering and Content-Based Recommendation")
    
    # Load data
    ratings, movies, tags, path = load_movielens_data()
    print(f"\nData loaded from: {path}")
    
    # Execute Question 1: Utility matrix and similarity
    utility_matrix = run_question1(ratings, movies)
    
    # Execute Question 2: Content-based recommendation
    movies_with_tags, tfidf_matrix, tfidf_vectorizer, feature_names = run_question2(movies, tags)
    
    print("\n" + "="*50)
    print("END OF PROJECT")
    print("="*50)

if __name__ == "__main__":
    main()
