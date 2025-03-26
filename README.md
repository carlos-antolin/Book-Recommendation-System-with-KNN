# Book Recommendation System with KNN

This project is part of the **Machine Learning with Python Certification** offered by **freeCodeCamp**. The objective is to create a book recommendation system using the K-Nearest Neighbors (KNN) algorithm with the Book-Crossings dataset.

## Project Overview
In this project, we:
- Load and preprocess the Book-Crossings dataset (1.1M ratings, 270K books, 90K users).
- Filter out users with fewer than 200 ratings and books with fewer than 100 ratings.
- Implement a KNN-based recommendation model using `NearestNeighbors` from `sklearn.neighbors`.
- Develop a function `get_recommends(book_title)` that suggests similar books.

## Dataset
The dataset contains book ratings from users. After preprocessing, it includes:
- Books with at least 100 ratings.
- Users who have rated at least 200 books.

## Requirements
- Python 3.x
- Scikit-learn
- Pandas
- NumPy
