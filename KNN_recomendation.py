# 1: Import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# 2: Get data files
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

!unzip book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# 3: Import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# 4: Filter books that have more than 100 ratings
book_count = df_ratings.groupby(["isbn"]).count().reset_index()

book_data = book_count.loc[book_count["rating"] >= 100]["isbn"]

book_data = df_books.loc[df_books["isbn"].isin(book_data)]

# 5: Filter users that have rated more than 200 books
user_count = df_ratings[["user", "rating"]].groupby(["user"]).count().reset_index()

users_data = user_count.loc[user_count["rating"] >= 200]["user"]

rating_data = df_ratings.loc[df_ratings["user"].isin(users_data)]

rating_data = rating_data.loc[rating_data["isbn"].isin(book_data["isbn"])]

# 6: Create data matrix

# Pivot the ratings into a user-book matrix
data_matrix = rating_data.pivot(index='isbn', columns='user', values='rating').fillna(0)

# Convert to a sparse matrix (for better optimization)
sparse_matrix = csr_matrix(data_matrix.values)

# 7: Create and train model

# Initialize K-Nearest Neighbors model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')

# Fit the model on the user-book matrix
model_knn.fit(sparse_matrix)

# 8: Function to return 5 recommended books based on the book title
def get_recommends(book=""):
    try:
        selected_book = book_data.loc[book_data["title"] == book]
    except KeyError as e:
        return ['The book does not exist.']

    selected_book_features = data_matrix.loc[data_matrix.index.isin(selected_book["isbn"])]

    # Calculate distances and indices of nearest neighbors
    distances, indices = model_knn.kneighbors([x for x in selected_book_features.values], n_neighbors=6)

    # Get the titles of the recommended books based on their indices
    recommended_titles = [
        df_books.loc[df_books['isbn'] == data_matrix.iloc[i].name]["title"].values[0]
        for i in indices[0][1:]
    ]

    return [book, [list(z) for z in zip(recommended_titles, distances[0][1:])][::-1]]

# 9: Test the model performance
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()
