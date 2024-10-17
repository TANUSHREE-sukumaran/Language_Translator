import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.preprocessing.text import Tokenizer


def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, sep='\s+', header=None)  
        print("Dataset loaded successfully.")
        
        print("Data contents:")
        print(data.head())  
        print(f"Data shape: {data.shape}")  
        
        
        if data.shape[1] < 3:
            print("Error: The dataset must have at least three columns (English, Hindi, French).")
            exit(1)

        return data[0].values, data[1].values, data[2].values  
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        exit(1)
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        exit(1)
    except pd.errors.ParserError:
        print("Error: The file could not be parsed. Please check the format.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading the dataset: {str(e)}")
        exit(1)


def create_model(vocab_size, output_size, max_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def preprocess_input(input_text, tokenizer, max_length):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_length, padding='post')
    return input_seq


class TranslationApp:
    def __init__(self, master):
        self.master = master
        master.title("Dual Language Translator")

        self.label = tk.Label(master, text="Enter English text:")
        self.label.pack()

        self.entry = tk.Entry(master)
        self.entry.pack()

        self.translate_button = tk.Button(master, text="Translate", command=self.translate)
        self.translate_button.pack()

        self.output_label = tk.Label(master, text="")
        self.output_label.pack()

    def translate(self):
        input_text = self.entry.get()
        if len(input_text) < 10:
            messagebox.showerror("Error", "Please provide a text with 10 or more letters.")
            return
        
        
        try:
            english_seq = preprocess_input(input_text, english_tokenizer, max_length)
            hindi_predictions = model_hindi.predict(english_seq)
            french_predictions = model_french.predict(english_seq)
            
            
            hindi_pred = np.argmax(hindi_predictions, axis=-1)[0]
            translated_hindi = hindi_tokenizer.sequences_to_texts([hindi_pred[hindi_pred != 0]])

            
            french_pred = np.argmax(french_predictions, axis=-1)[0]
            translated_french = french_tokenizer.sequences_to_texts([french_pred[french_pred != 0]])

            translated_hindi_text = translated_hindi[0] if translated_hindi else "No translation found."
            translated_french_text = translated_french[0] if translated_french else "No translation found."

            self.output_label.config(text=f"Hindi: {translated_hindi_text}\nFrench: {translated_french_text}")
        except Exception as e:
            messagebox.showerror("Translation Error", str(e))


english_words, hindi_words, french_words = load_dataset('d:/NULL CLASS/dual_translation/dictionary.txt')


english_tokenizer = Tokenizer()
hindi_tokenizer = Tokenizer()
french_tokenizer = Tokenizer()

english_tokenizer.fit_on_texts(english_words)
hindi_tokenizer.fit_on_texts(hindi_words)
french_tokenizer.fit_on_texts(french_words)


max_length = max(max(len(w.split()) for w in english_words), 10)  
X = english_tokenizer.texts_to_sequences(english_words)
X = pad_sequences(X, maxlen=max_length, padding='post')

y_hindi = hindi_tokenizer.texts_to_sequences(hindi_words)
y_hindi = pad_sequences(y_hindi, maxlen=max_length, padding='post')

y_french = french_tokenizer.texts_to_sequences(french_words)
y_french = pad_sequences(y_french, maxlen=max_length, padding='post')


y_hindi = np.expand_dims(y_hindi[:, 1:], axis=-1)  
y_french = np.expand_dims(y_french[:, 1:], axis=-1)  
X = X[:, :-1]  
vocab_size = len(english_tokenizer.word_index) + 1  
output_size_hindi = len(hindi_tokenizer.word_index) + 1  
model_hindi = create_model(vocab_size, output_size_hindi, max_length - 1)
model_hindi.fit(X, y_hindi, epochs=50, batch_size=32)


output_size_french = len(french_tokenizer.word_index) + 1 
model_french = create_model(vocab_size, output_size_french, max_length - 1)
model_french.fit(X, y_french, epochs=50, batch_size=32)


root = tk.Tk()
app = TranslationApp(root)
root.mainloop()
