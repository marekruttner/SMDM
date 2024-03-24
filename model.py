# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

# Load dataset
file_path = './dataset/test1000.csv'  # Make sure to replace this with the actual file path
dataset = pd.read_csv(file_path)

# Preprocess the data
X = dataset['Content'].fillna('')  # Replace NaN values with empty strings
y = dataset[['Likes', 'Analytics']]

# Tokenize and pad the "Content" text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded_sequences = pad_sequences(sequences, maxlen=1000, padding='post', truncating='post')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=256, input_length=1000),
    LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),  # L2 regularization
    Dropout(0.5),  # Dropout for regularization
    LSTM(64, kernel_regularizer=l2(0.001)),  # L2 regularization
    Dropout(0.5),  # Dropout for regularization
    Dense(64, activation='relu'),
    Dense(2)  # Output layer for 'Likes' and 'Analytics'
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.025))

# Train the model
model.fit(X_train, y_train, epochs=100, validation_split=0.2)

model.save('model/testlstm')

# Function to make predictions
def make_prediction(content):
    sequence = tokenizer.texts_to_sequences([content])
    padded = pad_sequences(sequence, maxlen=1000, padding='post', truncating='post')
    prediction = model.predict(padded)
    return prediction

# Example usage
#content_example = "V první třídě. - Děti, jaká je nejdůležitější tekutina pro život? ~ Benzín. Abychom si mohli dojet nakoupit!"

"""
Likes: 242
Analytics: 7.6K
"""
content_example = "aknsaknab dvMznejnfkaNvxĶNeta ifnkalsgnak"

"""
TRASH 
"""

prediction = make_prediction(content_example)
print(f"Predicted Likes and Analytics: {prediction}")
