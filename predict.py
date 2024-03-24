import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('./model/test1000_model_updated')

# Assuming there's a need to reverse the scaling for predictions
# This will depend on how you've applied scaling during training
# Initialize scaler instance (ensure to use the same parameters or load it if you've saved it)
scaler = MinMaxScaler()


# Function to prepare input text similarly to training data
def prepare_input(text):
    # This function should mirror the preprocessing done during training
    # For example, if you filled NaN values and converted texts to string, do it here too
    processed_text = text.fillna('')  # Replace NaN values with empty strings
    processed_text = processed_text.astype(str)  # Ensure data is treated as strings
    return processed_text


# Function to make predictions
def predict_likes_analytics(text):
    # Prepare the input text
    input_text = prepare_input(pd.Series([text]))  # Ensure the input is in the correct format

    # Make predictions
    predictions = model.predict(input_text)

    # Reverse the scaling of the predictions if necessary
    # predictions_scaled_back = scaler.inverse_transform(predictions) # Uncomment if scaling was used

    return predictions  # or predictions_scaled_back if scaling was reversed


# Example usage
text_to_predict = "Budoucnost je tady."
predictions = predict_likes_analytics(text_to_predict)
print(f"Predicted Likes and Analytics: {predictions}")
