import pytesseract

def extract_text(image):

    config = r'--oem 3 --psm 6'

    text = pytesseract.image_to_string(
        image,
        config=config
    )

    return text

