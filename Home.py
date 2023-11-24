import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.models import load_model

image_path = "DSC_0424-Edited.jpg"

st.set_page_config(layout="wide", page_title="Fake News Detector", page_icon = image_path)

padding = 20

st.title('Fake News Detector')

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def check_reality(news):
        model = joblib.load('model_lemmatization_svm_linear.pkl')
        model_lr = joblib.load('model.pkl')
    
        news = stemming(news)
    
        tfidf_vectorizer = joblib.load('tfidf_vectorizer_svm_lemma_linear.pkl')   
        tfidf_vectorizer_lr = joblib.load('tfidf_vectorizer.pkl')   
    
        input_features = tfidf_vectorizer.transform([news])
        input_features_lr = tfidf_vectorizer_lr.transform([news])
    
        prediction = model.predict(input_features)
        prediction_lr = model_lr.predict(input_features_lr)
    
        probs = model.predict_proba(input_features)
        probs_lr = model_lr.predict_proba(input_features_lr)

        probability_real = probs_lr[0][0]
        probability_fake = probs_lr[0][1]

        st.markdown("Results from model with SVM")
    
        if prediction[0] == 1:
            st.write("SVM say the news is fake")
        else:
            st.write("SVM say the news is real")
            
        if prediction_lr[0] == 1:
                pred = round(probability_fake*100)
                if pred < 71:
                        st.write("Model LR")
                        st.warning("Proceed with caution, as the certainty is low ⚠️")
                        st.markdown(
                        """
                        <style>
                                .stProgress > div > div > div > div {
                                background-image: linear-gradient(to right, #ffad60, #ffad60);
                                }
                        </style>""",
                        unsafe_allow_html=True,
                        )
                else:
                        st.write("Model LR")
                        st.markdown(
                        """
                        <style>
                                .stProgress > div > div > div > div {
                                background-image: linear-gradient(to right, #ff5733, #ffad60);
                                }
                        </style>""",
                        unsafe_allow_html=True,
                        )

                result = "The news is predicted to be " + str(pred) + "% fake 😬"
                st.progress(pred / 100)
                
        else:
                pred = round(probability_real*100)
                if pred < 70:
                        st.warning("Proceed with caution, as the certainty is low ⚠️")
                        st.markdown(
                        """
                        <style>
                                .stProgress > div > div > div > div {
                                background-image: linear-gradient(to right, #ffad60, #ffad60);
                                }
                        </style>""",
                        unsafe_allow_html=True,
                        )
                else:
                        st.markdown(
                                """
                                <style>
                                        .stProgress > div > div > div > div {
                                        background-image: linear-gradient(to right, #99ff99 , #00ccff);
                                        }
                                </style>""",
                                unsafe_allow_html=True,
                                )
                st.progress(pred / 100)
                result = "The news is predicted to be " + str(pred) + "% real 🥰" 
        return result

if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

col1, col2, col3 = st.columns([3,1,3])

with col1:
        st.write("Paste any news. We cannot guarantee the accuracy of our results, on top of this, make sure to do your own research also")
        st.write("To enter multiple news, put the word '//next news//'")
        text_input = st.text_area(
        "Paste your news 👇",
        placeholder="Enter your news here 📰",
        )
        proceed = st.button("Predict")
with col3:
        url_pattern = r'https?://\S+'

        if text_input or proceed:
                words = text_input.split()
                count = len(words)
                min_words = 30

                if count<min_words:
                        st.error("Error: Each news should have at least " + str(min_words) + " words. Currently it is " + str(count) + " words.")
                else:
                        news_segments = text_input.split("//next news//")
                        counter = 1
                        for segment in news_segments:
                                st.write("News " + str(counter) + ":")
                                counter += 1

                                count = len(segment)
                                min_words = 30

                                if count<min_words:
                                        st.error("Error: Each news should have at least " + str(min_words) + " words. Currently it is " + str(count) + " words.")
                                else:
                                        result = check_reality(segment)
                                        st.write(result)

# def main():
# if __name__ == "__main__":
        # main()
