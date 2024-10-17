import tkinter as tk
from datetime import datetime
import pandas as pd

def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, sep='\t', header=None, names=['English', 'Hindi'])
        return dict(zip(data['English'].str.lower(), data['Hindi']))
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        exit(1)

def translate_word():
    word = entry.get().strip().lower()
    
    if word[0] in 'aeiou':
        current_hour = datetime.now().hour
        if 21 <= current_hour < 22:
            translation = translation_dict.get(word, "Translation not found.")
            output_label.config(text=translation)
        else:
            output_label.config(text="This word starts with a vowel. Please provide another word.")
    else:
        translation = translation_dict.get(word, "Translation not found.")
        output_label.config(text=translation)

root = tk.Tk()
root.title("English to Hindi Translator")


translation_dict = load_dataset('dictionary.txt')


entry = tk.Entry(root, width=30)
entry.pack(pady=10)

translate_button = tk.Button(root, text="Translate", command=translate_word)
translate_button.pack(pady=5)

output_label = tk.Label(root, text="", wraplength=250)
output_label.pack(pady=20)


root.mainloop()
