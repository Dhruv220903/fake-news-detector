import streamlit as st
import joblib

vectorizer = joblib.load('vectorizer.pkl')
model= joblib.load('model.pkl')

st.title("Fake News Detector")
st.subheader("Enter the news article below to check whether it is fake or real:")

news_article = st.text_area("News Article")
if st.button("Check"):
    if news_article:
        # Preprocess the input text
        input_data = vectorizer.transform([news_article])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display the result
        if prediction == 1:
            st.success("The news article is REAL :).")
        else:
            st.error("The news article is FAKE!.")
    else:
        st.warning("Please enter a news article to check.")