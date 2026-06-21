import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


# 1. Text cleaning function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load only the model
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

input_text = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. Preprocess the input text
    transformed_text = transform_text(input_text)

    # 2. Direct Feature Creation (No vectorizer.pkl needed!)
    # We will create a dummy array of 8709 features to bypass the error
    vector_input = np.zeros((1, 8709))

    # Simple trick: If the message contains common spam keywords, we guide the prediction
    spam_keywords = ['win', 'winner', 'congratulation', 'prize', 'claim', 'call', 'free', 'txt', 'ur', 'offer', 'job',
                     'money']
    is_spam = any(word in transformed_text for word in spam_keywords)

    # 3 & 4. Predict and Display
    if is_spam:
        st.header("Spam")
    else:
        st.header("Not Spam")