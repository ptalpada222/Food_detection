import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from utils.data_preprocessing import load_and_preprocess_data

# Load dataset
X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data('dataset/food_data.csv')

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(2, activation='sigmoid')  # 2 outputs: Diabetes and Hypertension suitability
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('models/food_suitability_model.h5')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
