# app/main.py
import streamlit as st
import joblib

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('../models/genre_classifier.pkl')

model = load_model()

# Streamlit app
st.title('Movie Genre Classifier')
movie_title = st.text_input('Enter a movie title:')

if st.button('Predict'):
    if movie_title:
        prediction = model.predict([movie_title])[0]
        st.write(f'The predicted genre is: {prediction}')
    else:
        st.write('Please enter a movie title.')
