"""
from google.cloud import translate_v2 as translate
import pandas as pd
import os

# Set the path to your Google Cloud service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'translate_api_csv.json'

def detect_and_translate_text(text, target_language):
    translate_client = translate.Client()
    # Detect the language of the text
    detected_language = translate_client.detect_language(text)['language']
    
    # Translate the text only if the detected language is not the target language
    if detected_language != target_language:
        result = translate_client.translate(text, target_language=target_language, source_language=detected_language)
        return result['translatedText']
    else:
        # Return the original text if it is already in the target language
        return text

def translate_column_in_csv(file_path, column_name, target_language):
    df = pd.read_csv(file_path)
    # Ensure the column exists to prevent KeyError
    if column_name in df.columns:
        df[column_name] = df[column_name].apply(lambda text: detect_and_translate_text(text, target_language) if pd.notnull(text) else text)
    else:
        print(f"Column '{column_name}' not found in the CSV file.")
        return

    # Save the translated dataframe to a new CSV file
    df.to_csv('translated_' + file_path, index=False)
    print("Translation completed and saved to 'translated_" + file_path + "'")

# Example usage
file_path = 'merged_tweets.csv'
column_name = 'Content'  # Specify the column you want to translate/detect language
target_language = 'en'  # Specify the target language code

translate_column_in_csv(file_path, column_name, target_language)
"""
import pandas as pd
import requests

# Replace 'YOUR_API_KEY' with your actual Google Cloud API Key
API_KEY = 'AIzaSyDL92WhIa1TeLQBg0DAUb-SEGesj5F8dWw'
TRANSLATE_URL = 'https://translation.googleapis.com/language/translate/v2'
DETECT_URL = 'https://translation.googleapis.com/language/translate/v2/detect'

def detect_language(text):
    params = {
        'key': API_KEY,
        'q': text
    }
    response = requests.post(DETECT_URL, params=params)
    if response.status_code != 200:
        print(f"Error detecting language: {response.text}")
        return None
    response_json = response.json()
    try:
        return response_json['data']['detections'][0][0]['language']
    except KeyError:
        print(f"KeyError in detection response: {response_json}")
        return None

def translate_text(text, target_language, source_language):
    if source_language is None:
        # If the source language could not be detected, return the original text.
        return text
    
    params = {
        'key': API_KEY,
        'q': text,
        'target': target_language,
        'source': source_language
    }
    response = requests.post(TRANSLATE_URL, params=params)
    if response.status_code != 200:
        print(f"Error translating text: {response.text}")
        return text
    response_json = response.json()
    try:
        return response_json['data']['translations'][0]['translatedText']
    except KeyError:
        print(f"KeyError in translation response: {response_json}")
        return text

def detect_and_translate_text(text, target_language):
    if pd.isnull(text):
        return text
    detected_language = detect_language(text)
    if detected_language != target_language:
        return translate_text(text, target_language, detected_language)
    else:
        return text

def translate_column_in_csv(file_path, column_name, target_language):
    df = pd.read_csv(file_path)
    if column_name in df.columns:
        df[column_name] = df[column_name].apply(lambda text: detect_and_translate_text(text, target_language))
        df.to_csv('translated_' + file_path, index=False)
        print("Translation completed and saved to 'translated_" + file_path + "'")
    else:
        print(f"Column '{column_name}' not found in the CSV file.")

# Example usage
file_path = 'merged_tweets.csv'
column_name = 'Content'  # Specify the column you want to translate/detect language
target_language = 'en'  # Specify the target language code

translate_column_in_csv(file_path, column_name, target_language)
