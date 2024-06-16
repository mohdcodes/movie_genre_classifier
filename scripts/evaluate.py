# scripts/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['title', 'genre'])
    return data

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    test_data = load_data('../data/test_data.txt')
    solution_data = load_data('../data/test_data_solution.txt')
    X_test = test_data['title']
    y_test = solution_data['genre']

    model = joblib.load('../models/genre_classifier.pkl')
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Model Accuracy: {accuracy:.2f}')
