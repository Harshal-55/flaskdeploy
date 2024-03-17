import numpy as np
import nltk
import string
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from fastapi import FastAPI

app = FastAPI()

# Define your FastAPI routes and logic below...

# Load environment variables from .env file
load_dotenv()

# Function to read text from file
def read_file(file_path):
    try:
        with open(file_path, 'r', errors='ignore') as f:
            raw_doc = f.read()
        return raw_doc.lower()
    except FileNotFoundError:
        print("File not found. Please enter a valid file path.")
        return None

# Function to greet the user
def greet(sentence):
    greet_inputs = ('hello', 'hi', 'wassup', 'how are you?')
    greet_responses = ('hi', 'hey', 'There there!', 'hey there!')
    if sentence.lower() in greet_inputs:
        return random.choice(greet_responses)

# Function to lemmatize tokens
def LemTokens(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

# Function to normalize text
def LemNormalize(text):
    remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Main function to generate bot responses
def response(user_response, sentence_tokens, raw_doc):
    # Add user response to sentence tokens
    sentence_tokens.append(user_response)
    
    # TF-IDF vectorization
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    
    # Calculate cosine similarity between user response and sentences in file
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]
    
    # If no matching sentence found, return default response
    if req_tfidf == 0:
        return "I'm sorry, I couldn't understand your question."
    else:
        return raw_doc.split('.')[idx]

if __name__ == "__main__":
    file_path = os.getenv("FILE_PATH")
    raw_doc = read_file(file_path)

    if raw_doc:
        # Tokenization and other preprocessing steps
        nltk.download('punkt')
        nltk.download('wordnet')
        sentence_tokens = nltk.sent_tokenize(raw_doc)
        
        flag = True
        print('Hello! I am the Learning bot. Start typing your text after greeting to talk to me. To end the conversation, type Bye!')
        
        while flag:
            user_response = input().lower()
            
            if user_response != 'bye':
                if user_response == 'thank you' or user_response == 'thanks':
                    flag = False
                    print('Bot: You are Welcome...')
                else:
                    if greet(user_response) is not None:
                        print('Bot:', greet(user_response))
                    else:
                        print('Bot:', response(user_response, sentence_tokens, raw_doc))
            else:
                flag = False
                print('Bot: Goodbye!')
