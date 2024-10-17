import speech_recognition as sr
import pyttsx3
from transformers import MarianMTModel, MarianTokenizer
import keyboard  
import threading  
import time 
stop_translation = False

def load_dataset(file_path):
    source_texts, target_texts = [], []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()

        for line in data:
            # Strip whitespace and split by tab
            parts = line.strip().split('\t')
            # Ensure there are exactly two parts
            if len(parts) == 2:
                source_texts.append(parts[0].strip())  
                target_texts.append(parts[1].strip())
            else:
                print(f"Skipping line due to incorrect format: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        exit(1)

    return source_texts, target_texts

def translate_text(text, src_lang, dest_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{dest_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True, truncation=True))
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translation

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def capture_voice(language='en-US'):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language=language)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None

def monitor_keyboard():
    global stop_translation
    while True:
        if keyboard.is_pressed('5'):  # Check if 5 is pressed
            print("Stopping translation...")
            stop_translation = True
            break
        time.sleep(0.1)  

def main():
    keyboard_thread = threading.Thread(target=monitor_keyboard)
    keyboard_thread.start()
    source_texts, target_texts = load_dataset('d:/NULL CLASS/voice.txt') 

    while not stop_translation:
        english_input = capture_voice(language='en-US')
        if english_input:
            translated_to_spanish = translate_text(english_input, src_lang='en', dest_lang='es')
            print(f"Translated to Spanish: {translated_to_spanish}")  
            speak_text(translated_to_spanish)
        time.sleep(0.1)  

if __name__ == "__main__":
    main()
