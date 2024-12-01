""" Install necessary packages if you haven't already using pip install command 
kaggle
gensim
scikit-learn
nltk
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/BA_AirlineReviews.csv')

# Display the first few rows of the dataset
print(df.head())

# Preprocess the text data by removing stopwords and non-alphanumeric characters
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_reviews'] = df['ReviewBody'].apply(preprocess_text)

# Display the cleaned reviews
print(df[['ReviewBody', 'cleaned_reviews']].head())

# Map ratings to sentiments
def map_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['OverallRating'].apply(map_sentiment)

# Features (X) and target labels (y)
X = df['cleaned_reviews']
y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Use CountVectorizer for bag-of-words representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train a logistic regression model on the bag-of-words features
model = LogisticRegression()
model.fit(X_train_bow, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_bow)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Train a Word2Vec model for the reviews
sentences = [review.split() for review in X_train]
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Function to get the average vector of a review
def get_vector(review):
    words = review.split()
    vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# Transform the training and testing data using Word2Vec
X_train_w2v = np.array([get_vector(review) for review in X_train])
X_test_w2v = np.array([get_vector(review) for review in X_test])

# Standardize the Word2Vec features
scaler = StandardScaler()
X_train_w2v = scaler.fit_transform(X_train_w2v)
X_test_w2v = scaler.transform(X_test_w2v)

# Train the logistic regression model on the Word2Vec features
model = LogisticRegression(max_iter=200)
model.fit(X_train_w2v, y_train)

# Predict and evaluate the model on Word2Vec features
y_pred = model.predict(X_test_w2v)
print('Accuracy (Word2Vec):', accuracy_score(y_test, y_pred))
print('Classification Report (Word2Vec):\n', classification_report(y_test, y_pred))
