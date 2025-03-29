import joblib
import tensorflow as tf
from utils.data_preprocessing import clean_text
import numpy as np
from gtts import gTTS
import os

# Load trained model and vectorizer
model = tf.keras.models.load_model('models/food_suitability_model.h5')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict_suitability(ingredients):
    ingredients_cleaned = clean_text(ingredients)
    ingredients_vectorized = vectorizer.transform([ingredients_cleaned])

    prediction = model.predict(ingredients_vectorized)
    diabetes_risk = "Not suitable" if prediction[0][0] < 0.5 else "Suitable"
    hypertension_risk = "Not suitable" if prediction[0][1] < 0.5 else "Suitable"

    return {
        "Diabetes": diabetes_risk,
        "Hypertension": hypertension_risk
    }

#for text to speech 

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")  # Plays audio on Linux

def generate_voice_feedback(result):
    feedback = f"For Diabetes: {result['Diabetes']}. For Hypertension: {result['Hypertension']}."
    text_to_speech(feedback)


# Example Prediction
print(predict_suitability("sugar, salt, milk, almonds"))
