# Fake News Detector

Deployment at: https://fake-news-detector-nicholas.streamlit.app/

## Overview
Fake News Detector is a Natural Language Processing (NLP) project aimed at distinguishing between fake and real news articles. The project involves web scraping datasets from various news websites to collect data of both fake and real news articles. Using techniques such as lemmatization and stemming, in combination with scikit-learn's Logistic Regression (LR) and Support Vector Machine (SVM) algorithms, the project aims to develop an optimized model for detecting fake news articles.

## Key Points
- **Web Scraping:** A total of 44,898 news (23,481 fake and 21,417 true news) were collected in order to create a comphrehensive dataset
- **Data Preprocessing:** Utilizes lemmatization and stemming techniques to preprocess text data and extract meaningful features.
- **Machine Learning Models:** Implements scikit-learn's LR and SVM algorithms to classify news articles as fake or real.
- **Model Optimization:** Explores various strategies to optimize model performance, including parameter tuning and feature selection.

## Technologies Used
- **Python:** Primary programming language for the project.
- **Natural Language Processing (NLP) Libraries:** Utilizes NLTK (Natural Language Toolkit) and other NLP libraries for text preprocessing.
- **Scikit-learn:** Implements machine learning models such as Logistic Regression and Support Vector Machine.
- **Web Scraping Tools:** Utilizes BeautifulSoup for web scraping news articles.
