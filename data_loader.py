#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import kagglehub

def load_movielens_data():
    """Downloads and loads the MovieLens dataset"""
    path = kagglehub.dataset_download("grouplens/movielens-latest-small")
    print("Path to dataset files:", path)
    
    # Load data files
    ratings_path = os.path.join(path, "ratings.csv")
    movies_path = os.path.join(path, "movies.csv")
    tags_path = os.path.join(path, "tags.csv")
    
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    tags = pd.read_csv(tags_path)
    
    return ratings, movies, tags, path 