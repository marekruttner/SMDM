import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('./dataset/test1000.csv')

# Preprocessing to ensure 'Content' is properly formatted
data['Content'] = data['Content'].fillna('')  # Replace NaN values with empty strings
data['Content'] = data['Content'].astype(str)  # Ensure all data is treated as strings

# Select 'Content' as input
X = data['Content']
# Assuming 'Likes' and 'Analytics' are targets
y = data[['Likes', 'Analytics']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize targets
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

# Setup TextVectorization layer
max_features = 10000
sequence_length = 250

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Adapt the vectorization layer to the training text data
vectorize_layer.adapt(X_train)

# Updated Model definition with an explicit input layer for string data
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),  # Explicit input shape for strings
    vectorize_layer,
    tf.keras.layers.Embedding(max_features + 1, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)  # For 'Likes' and 'Analytics'
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_train, y_train_scaled, validation_split=0.2, epochs=100)

# Evaluate model
model.evaluate(X_test, y_test_scaled)

# Save model
model.save('./model/test1000_model')
