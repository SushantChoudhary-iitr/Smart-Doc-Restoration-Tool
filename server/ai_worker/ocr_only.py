import sys
import cv2
import pytesseract
from preprocess import preprocess_image
from ocr import extract_text
from strip import clean_text
from restore_text import restore_text

image_path = sys.argv[1]

processed_image = preprocess_image(image_path)

text = extract_text(processed_image)

cleaned_text = clean_text(text)

restored_text = restore_text(cleaned_text)

print(restored_text)