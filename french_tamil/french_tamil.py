import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import messagebox


data = pd.read_csv('french_tamil.csv')


data = data[data['French'].apply(lambda x: len(x) == 5)]


french_words = data['French'].values
tamil_words = data['Tamil'].values


tokenizer_french = Tokenizer()
tokenizer_french.fit_on_texts(french_words)
french_sequences = tokenizer_french.texts_to_sequences(french_words)

tokenizer_tamil = Tokenizer()
tokenizer_tamil.fit_on_texts(tamil_words)
tamil_sequences = tokenizer_tamil.texts_to_sequences(tamil_words)


max_french_length = max(len(seq) for seq in french_sequences)
french_padded = pad_sequences(french_sequences, maxlen=max_french_length, padding='post')


tamil_padded = pad_sequences(tamil_sequences, padding='post')
tamil_padded = [item for sublist in tamil_padded for item in sublist if item != 0]  
tamil_padded = np.array(tamil_padded).reshape(-1, 1)  
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer_french.word_index) + 1, output_dim=64, input_length=max_french_length))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(len(tokenizer_tamil.word_index) + 1, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(french_padded, np.array(tamil_padded), epochs=50, validation_split=0.2)

def translate():
    french_word = entry.get()
    if len(french_word) != 5:
        messagebox.showinfo("Error", "Please enter a 5-letter French word.")
        return
    
    sequence = tokenizer_french.texts_to_sequences([french_word])
    padded_sequence = pad_sequences(sequence, maxlen=max_french_length, padding='post')
    predicted = model.predict(padded_sequence)
    predicted_index = np.argmax(predicted, axis=-1)
    
    translated_word = tokenizer_tamil.index_word.get(predicted_index[0], "Translation not found")
    output_label.config(text=f'Tamil Translation: {translated_word}')


window = tk.Tk()
window.title("French to Tamil Translator")

entry = tk.Entry(window)
entry.pack(pady=10)

translate_button = tk.Button(window, text="Translate", command=translate)
translate_button.pack(pady=10)

output_label = tk.Label(window, text="")
output_label.pack(pady=10)

window.mainloop()
