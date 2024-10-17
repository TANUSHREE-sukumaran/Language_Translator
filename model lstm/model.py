import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

# Create a simple example dataset
data = pd.DataFrame({
    'source_column': ['hello', 'world'],  # Replace with actual sentences
    'target_column': ['hola', 'mundo']     # Replace with actual translations
})

# Tokenization
tokenizer_src = Tokenizer()
tokenizer_tgt = Tokenizer()

# Fit the tokenizers on the dataset
tokenizer_src.fit_on_texts(data['source_column'])
tokenizer_tgt.fit_on_texts(data['target_column'])

# Convert texts to sequences
input_sequences = tokenizer_src.texts_to_sequences(data['source_column'])
target_sequences = tokenizer_tgt.texts_to_sequences(data['target_column'])

# Padding
max_input_length = max(len(seq) for seq in input_sequences)
max_target_length = max(len(seq) for seq in target_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_target_length, padding='post')

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer_src.word_index) + 1, output_dim=256),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(tokenizer_tgt.word_index) + 1, activation='softmax'))
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(input_sequences, np.expand_dims(target_sequences, -1), epochs=10, batch_size=1)

# Save the model
model.save('your_model.h5')

# Function to translate a sentence
def translate(sentence):
    # Convert the input sentence to a sequence
    sequence = tokenizer_src.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_input_length, padding='post')

    # Predict the target sequence
    predicted = model.predict(padded_sequence)
    predicted_sequence = np.argmax(predicted, axis=-1)

    # Convert the predicted sequence back to words
    translated_sentence = ' '.join(tokenizer_tgt.index_word[i] for i in predicted_sequence[0] if i != 0)
    
    return translated_sentence

# Example usage of the translate function
print(translate('hello'))  # Should output 'hola'
