import time

import cv2
import numpy as np
import pytesseract
import easyocr


def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Optional: Dilate to reduce noise
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    return dilated


# Main function to extract text
def extract_text_tesseract(preprocessed_img):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed_img, config=custom_config)
    return text


def extract_text_easyocr(preprocessed_img):
    reader = easyocr.Reader(['en'], gpu=True, quantize=True)
    result = reader.readtext(image_path, detail=0)
    return ' '.join(result)


if __name__ == "__main__":
    image_path = 'cart1.jpg'
    preprocessed_img = preprocess_image(image_path)

    start_time = time.time()
    extracted_text_tesseract = extract_text_tesseract(preprocessed_img)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(extracted_text_tesseract)
    print('Elapsed time: ', elapsed_time)

    start_time = time.time()
    extracted_text_easyocr = extract_text_easyocr(preprocessed_img)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(extracted_text_easyocr)
    print('Elapsed time: ', elapsed_time)



