from flask import Flask, render_template, request, jsonify
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
import pandas as pd
import os
from transformers import pipeline
import wikipedia
import random

# Initialize Flask app
app = Flask(__name__)

# Load AI model for fallback responses
chat_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Initialize ChatterBot
chatbot = ChatBot(
    'EduBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///db.sqlite3',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter'
    ]
)

# Memory storage for personalization
memory = {"name": None, "age": None}

# Load dataset from CSV
csv_path = r"educational/Dataset_Python_Question_Answer.csv"
qa_dict = {}
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path).dropna().astype(str)
    for index, row in data.iterrows():
        question, answer = row.iloc[0].strip().lower(), row.iloc[1].strip()
        qa_dict[question] = answer

# Basic conversation responses
basic_responses = {
    "hello": "Hello! How can I assist you today? 😊",
    "hi": "Hi there! Ready to learn something new? 📚",
    "hey": "Hey! How can I help you today?",
    "how are you": "I'm just a chatbot, but I'm here to help you learn! 📖",
    "who are you": "I am EduBot🤖, your AI learning assistant!",
    "what is your name": "My name is EduBot, but you can call me Edu! 🤖",
    "what is the capital of pakistan?": "The capital of Pakistan is Islamabad. 🇵🇰",
    "who is albert einstein?": "Albert Einstein was a brilliant physicist known for his theories on relativity and the photoelectric effect. 🧠",
}

# Function to fetch Wikipedia summaries
def get_wiki_summary(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        return None  # Wikipedia couldn't find an answer

# Function to generate chatbot responses
def chat_with_memory(user_input):
    user_input = user_input.lower().strip()
    # Check predefined basic responses first
    if user_input in basic_responses:
        return basic_responses[user_input]
    # Check if the question exists in the dataset
    if user_input in qa_dict:
        return qa_dict[user_input]
    # Handle name and age storage
    if "my name is" in user_input:
        name = user_input.split("my name is ")[1].strip()
        if name:
            memory["name"] = name
            return f"Nice to meet you, {name}! 😊"
        return "I didn't catch your name. Can you repeat it?"
    if "my age is" in user_input or ("i am" in user_input and "years old" in user_input):
        age = ''.join(filter(str.isdigit, user_input))
        if age:
            memory["age"] = age
            return f"Wow, {age} years old! That's great!"
        return "I couldn't understand your age. Please enter a number."
    if "what is my age" in user_input:
        return f"You are {memory['age']} years old!" if memory["age"] else "I don't know your age yet!"
    # Try Wikipedia-based answers
    wiki_response = get_wiki_summary(user_input)
    if wiki_response:
        return wiki_response
    # Use GPT model if Wikipedia fails
    gpt_response = chat_pipeline(user_input, max_length=100, pad_token_id=50256)
    gpt_text = gpt_response[0]['generated_text']
    if gpt_text.lower() != user_input and len(gpt_text.split()) > 3:
        return gpt_text
    # Use ChatterBot if GPT fails
    response = chatbot.get_response(user_input)
    if response.confidence < 0.5 or response.text.lower() == user_input:
        return random.choice([
            "I'm not sure about that. Can you ask in a different way?",
            "I couldn't find an answer. Maybe try rephrasing?",
            "That's a tough one! Let me learn more and get back to you.",
            "I'm still learning! Can you provide more details?",
        ])
    return response.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response = chat_with_memory(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
