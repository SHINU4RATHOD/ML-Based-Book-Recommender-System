# Importing necessary libraries for data handling
import pandas as pd
import numpy as np

# Loading books data with custom settings to handle special characters
books = pd.read_csv('data/BX-Books.csv', sep=";", on_bad_lines='skip', encoding='latin-1')

# Display first few rows to see what the data looks like
books.head()

# Show information about the books dataset (like column names and types)
books.info()

# Display the large image URL for a specific book
books.iloc[237]['Image-URL-L']

# Keep only columns that matter (like title, author, year, etc.)
books = books[['ISBN','Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-L']]

# Rename columns to simpler names for easy access later
books.rename(columns={"Book-Title":'title',
                      'Book-Author':'author',
                      "Year-Of-Publication":'year',
                      "Publisher":"publisher",
                      "Image-URL-L":"image_url"}, inplace=True)

# Check out how the updated books data looks
books.head()

# Now, let's work with user data
users = pd.read_csv('data/BX-Users.csv', sep=";", on_bad_lines='skip', encoding='latin-1')

# Rename columns to make them easy to read
users.rename(columns={"User-ID":'user_id',
                      'Location':'location',
                     "Age":'age'}, inplace=True)
users.head()

# Now, let's load the ratings data
ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=";", on_bad_lines='skip', encoding='latin-1')

# Rename columns to simpler names for easy access
ratings.rename(columns={"User-ID":'user_id',
                      'Book-Rating':'rating'}, inplace=True)
ratings.head()


# =============================================================================
# # ### Conclution:
# # Now we have 3 dataframes
#  
# # * books
# # * users
# # * ratings
# 
# =============================================================================

# Now we have 3 datasets: books, users, and ratings
print(books.shape, users.shape, ratings.shape, sep='\n')

# Find users who rated more than 200 books (they are 'active' users)
x = ratings['user_id'].value_counts() > 200
y = x[x].index  # Get only those users who rated 200+ books

# Filter ratings data to keep only active users
ratings = ratings[ratings['user_id'].isin(y)]

# Combine ratings and books data so we can work with all info together
ratings_with_books = ratings.merge(books, on='ISBN')
ratings_with_books.head()

# Calculate the number of ratings each book received
number_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns={'rating':'num_of_rating'}, inplace=True)

# Add the number of ratings info to our main dataset
final_rating = ratings_with_books.merge(number_rating, on='title')

# Keep only books with at least 50 ratings
final_rating = final_rating[final_rating['num_of_rating'] >= 50]

# Remove duplicate entries to keep only unique user-book pairs
final_rating.drop_duplicates(['user_id','title'], inplace=True)

# Create a pivot table with books as rows, users as columns, and ratings as values
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating')
book_pivot.fillna(0, inplace=True)  # Fill missing values with 0
book_pivot.index[3]    
#keeping books name
book_names = book_pivot.index
book_names[2]
np.where(book_pivot.index == '4 Blondes')[0][0]

# final_rating['title'].value_counts()
ids = np.where(final_rating['title'] == "Harry Potter and the Chamber of Secrets (Book 2)")[0][0]
ids


# To find similar books, we'll make this data "sparse" for efficient calculation
from scipy.sparse import csr_matrix
book_sparse = csr_matrix(book_pivot)

######################################## model building and training
# Using Nearest Neighbors algorithm to find similar books (this is a type of clustering)
from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm= 'brute')
model.fit(book_sparse)  # Train model to find nearest neighbors (similar books)

# Find similar books to a specific book based on ratings
distance, suggestion = model.kneighbors(book_pivot.iloc[237,:].values.reshape(1,-1), n_neighbors=6)

# See book titles of suggestions
for i in range(len(suggestion)):
    print(book_pivot.index[suggestion[i]])
    
   
    
book_name = []
for book_id in suggestion:
    book_name.append(book_pivot.index[book_id])

book_name


ids_index = []
for name in book_name[0]: 
    ids = np.where(final_rating['title'] == name)[0][0]
    ids_index.append(ids)
    
    
for idx in ids_index:
    url = final_rating.iloc[idx]['image_url']
    print(url)

    
# # Saving our work so we can use it later without recalculating everything
# import pickle
# pickle.dump(model, open('artifacts/model.pkl','wb'))
# pickle.dump(book_names, open('artifacts/book_names.pkl','wb'))
# pickle.dump(final_rating, open('artifacts/final_rating.pkl','wb'))
# pickle.dump(book_pivot, open('artifacts/book_pivot.pkl','wb'))



# Function to recommend books based on a given book name
def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)
    
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            if j == book_name:
                print(f"You searched '{book_name}'\n")
                print("The suggestion books are: ")
            else:
                print(j)

# Test recommendation function with a specific book
book_name = "Harry Potter and the Chamber of Secrets (Book 2)"
recommend_book(book_name)
