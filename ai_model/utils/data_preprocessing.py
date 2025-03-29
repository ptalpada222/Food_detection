import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Ingredients'] = df['Ingredients'].apply(clean_text)

    # Convert textual labels to numerical
    df['Suitable for Diabetes'] = df['Suitable for Diabetes'].map({'Yes': 1, 'No': 0})
    df['Suitable for Hypertension'] = df['Suitable for Hypertension'].map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(df['Ingredients'], df[['Suitable for Diabetes', 'Suitable for Hypertension']], test_size=0.2, random_state=42)

    # Convert text into numerical features
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test, vectorizer
