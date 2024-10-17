import pandas as pd
import numpy as np
import cv2
import pytesseract
from tkinter import Tk, Button, filedialog, Text
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract."""
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        extracted_text = pytesseract.image_to_string(img_rgb)
        return extracted_text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def translate_text(text):
    """Translate extracted text to the desired language."""
    return text

def process_image():
    """Handle the image upload and processing."""
    file_path = filedialog.askopenfilename(title="Select an Image",
                                          filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        extracted_text = extract_text_from_image(file_path)
        if extracted_text:
            translated_text = translate_text(extracted_text)
            output_text.delete(1.0, 'end')  
            output_text.insert('end', f"Extracted Text: {extracted_text}\n")
            output_text.insert('end', f"Translated Text: {translated_text}\n")
        else:
            output_text.insert('end', "Error extracting text from the image.\n")

root = Tk()
root.title("Image Text Extraction and Translation")

upload_button = Button(root, text="Upload Image", command=process_image)
upload_button.pack(pady=10)

output_text = Text(root, wrap='word', width=50, height=15)
output_text.pack(pady=10)

root.mainloop()
