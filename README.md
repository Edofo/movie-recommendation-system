# Collaborative Filtering and Content-Based Recommendation

This project implements two recommendation system approaches using the MovieLens Latest Small dataset:
1. Collaborative filtering based on a utility matrix
2. Content-based recommendation with TF-IDF

## Project Structure

The project is organized into several modules for better separation of concerns:

- **main.py**: Main entry point that orchestrates the execution of different parts
- **data_loader.py**: Module for downloading and loading MovieLens data
- **question1.py**: Implementation of utility matrix and similarity calculations (cosine, Pearson)
- **question2.py**: Implementation of a content-based recommendation system with TF-IDF

## Dependencies Installation

```bash
pip install -r requirements.txt
```

## Execution

To run the project, simply execute:

```bash
python main.py
```

## Features

### Question 1: Utility Matrix and Similarity
- Creation of a utility matrix with users as rows and movies as columns
- Calculation of similarity between users and movies using:
  - Cosine similarity
  - Pearson correlation
- Interpretation of similarity results

### Question 2: Content-Based Recommendation
- Data preprocessing (titles, genres, tags)
- TF-IDF representation of movies
- Recommendation of similar movies to a given movie
- Visualization of important features
- Discussion of advantages and limitations of the approach

## Analysis and Results

The program displays various information, including:
- Utility matrix density
- Similarities between selected users and movies
- Content-based recommendations for a chosen movie
- Visualization of the most important terms for a movie
- Analysis of the advantages and disadvantages of the approaches used