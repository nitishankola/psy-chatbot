from flask import Flask, render_template, request, jsonify
import joblib
from nltk.stem import WordNetLemmatizer
import nltk
import random
import requests
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load the model
patterns_dict, responses_dict = joblib.load('psychology_chatbot_model.joblib')

# Initialize the lemmatizer and create a set of stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Function to get a response from the chatbot
def get_response(user_input):
    user_input_tokens = preprocess_text(user_input)
    for pattern, tag in patterns_dict.items():
        if set(user_input_tokens).intersection(set(pattern)):
            return random.choice(responses_dict[tag])
    return get_gemini_response(user_input)

# Configure the Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=api_key)

# Function to call Gemini API
def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(question)
    if response:
        return response.text
    else:
        return "I'm sorry, I don't understand."

# Flask routes
@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/bot')
def bot():
    return render_template('bot.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_input = request.form['user_input']
    response = get_response(user_input)
    return jsonify({'bot_response': response})

if __name__ == '__main__':
    app.run(debug=True)
