# Collaborative Filtering and Content-Based Recommendation

This project implements two recommendation system approaches using the MovieLens Latest Small dataset:
1. Collaborative filtering based on a utility matrix
2. Content-based recommendation with TF-IDF

The system is evaluated using several offline metrics to assess recommendation quality.

## Project Structure

The project is organized into several modules for better separation of concerns:

- **main.py**: Main entry point that orchestrates the execution of different parts
- **data_loader.py**: Module for downloading and loading MovieLens data
- **question1.py**: Implementation of utility matrix and similarity calculations (cosine, Pearson)
- **question2.py**: Implementation of a content-based recommendation system with TF-IDF
- **evaluation.py**: Implementation of offline evaluation metrics

## Dependencies Installation

```bash
pip install -r requirements.txt
```

## Execution

To run the project without evaluation (faster):

```bash
python main.py
```

To run with evaluation metrics:

```bash
python main.py --evaluate
```

You can optimize the evaluation performance with these parameters:

```bash
# Evaluate with only 20 users (faster)
python main.py --evaluate --max-users 20

# Limit RMSE calculation to 5000 samples (much faster)
python main.py --evaluate --max-samples 5000

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

### Evaluation Metrics
- **RMSE (Root Mean Square Error)**: Measures the accuracy of rating predictions
- **Precision, Recall, F1-score**: Assesses the relevance of recommended items
- **MAP (Mean Average Precision)**: Evaluates the ranking quality with binary relevance
- **NDCG (Normalized Discounted Cumulative Gain)**: Measures the ranking quality with relevance grading

## Analysis and Results

The program displays various information, including:
- Utility matrix density
- Similarities between selected users and movies
- Content-based recommendations for a chosen movie
- Visualization of the most important terms for a movie
- Analysis of the advantages and disadvantages of the approaches used
- Evaluation results showing the performance of the recommendation system

## Troubleshooting

If the evaluation metrics take too long to calculate:

1. Reduce the number of users evaluated: `python main.py --evaluate --max-users 5`
2. Reduce the number of samples for RMSE: `python main.py --evaluate --max-samples 1000`
3. Run without evaluation to see just the recommendation system: `python main.py`