import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()


# Function to transform text
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


# Load the vectorizer and model
tfidf = pickle.load(open('sms_spam_detection_vectorizer.pkl', 'rb'))
model = pickle.load(open('sms_spam_detection_model.pkl', 'rb'))

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input from user
input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    st.write("Transformed Text:", transformed_sms)  # Debugging output

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    st.write("Vector Input Shape:", vector_input.shape)  # Debugging output

    # Predict
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
