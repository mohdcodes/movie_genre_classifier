# scripts/train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['title', 'genre'])
    return data

# Preprocess and split data
def preprocess_data(data):
    X = data['title']
    y = data['genre']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
def train_model(X_train, y_train):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model

# Save model
def save_model(model, file_path):
    joblib.dump(model, file_path)

if __name__ == "__main__":
    train_data = load_data('../data/train.txt')
    X_train, X_val, y_train, y_val = preprocess_data(train_data)
    model = train_model(X_train, y_train)
    save_model(model, '../models/genre_classifier.pkl')
