import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import load_dataset


# Load the Twitter Sentiment Analysis dataset
dataset = load_dataset('carblacac/twitter-sentiment-analysis')
df = dataset['train'].to_pandas()


# Assuming the target column is named 'label' (replace with the correct column name)
text_column = 'text'  # Replace with the correct column name
target_column = 'feeling'  # Replace with the correct column name


# Ensure 'label' is categorical
label_encoder = LabelEncoder()
df['feeling'] = label_encoder.fit_transform(df['feeling'])


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df[text_column], df[target_column], test_size=0.2, random_state=42)


# Tokenize and pad sequences for input to Bi-LSTM
max_words = 10000  # Set the maximum number of words in your vocabulary
max_len = 100  # Set the maximum length of your sequences


tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)


X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post')
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding='post')


# Bi-LSTM model
embedding_dim = 50  # Set the embedding dimension
hidden_units = 64  # Set the number of hidden units in the LSTM layer


model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    Bidirectional(LSTM(hidden_units, return_sequences=True)),
    Bidirectional(LSTM(hidden_units)),
    Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Display the model summary
model.summary()


# Train the model
epochs = 5  # Set the number of training epochs
batch_size = 32  # Set the batch size
model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)


# Evaluate the model
loss, accuracy = model.evaluate(pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding='post'), y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
 
