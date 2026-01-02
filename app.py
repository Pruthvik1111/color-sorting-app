import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("sorter_model.h5")

IMG_SIZE = 128

# Must match training class order
class_names = ['blue', 'green', 'red']

st.set_page_config(page_title="Color Sorting AI", layout="centered")

st.title("🎨 Color Sorting AI System")
st.write("Upload an image and the model will predict the dominant color.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image
def detect_color_hsv(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Color ranges
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = red1 + red2

    green_mask = cv2.inRange(hsv, (36, 50, 50), (89, 255, 255))
    blue_mask = cv2.inRange(hsv, (90, 50, 50), (128, 255, 255))

    red_pixels = red_mask.sum()
    green_pixels = green_mask.sum()
    blue_pixels = blue_mask.sum()

    total = red_pixels + green_pixels + blue_pixels
    if total == 0:
        return "Unknown", 0

    if red_pixels > green_pixels and red_pixels > blue_pixels:
        return "Red", (red_pixels / total) * 100
    elif green_pixels > blue_pixels:
        return "Green", (green_pixels / total) * 100
    else:
        return "Blue", (blue_pixels / total) * 100

def detect_color_hsv(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    color_ranges = {
        "Red": [
            ((0, 70, 50), (10, 255, 255)),
            ((170, 70, 50), (180, 255, 255))
        ],
        "Orange": [((11, 70, 50), (25, 255, 255))],
        "Yellow": [((26, 70, 50), (35, 255, 255))],
        "Green": [((36, 50, 50), (85, 255, 255))],
        "Cyan": [((86, 50, 50), (95, 255, 255))],
        "Blue": [((96, 50, 50), (125, 255, 255))],
        "Purple": [((126, 50, 50), (145, 255, 255))],
        "Pink": [((146, 50, 50), (169, 255, 255))],
        "Black": [((0, 0, 0), (180, 255, 50))],
        "White": [((0, 0, 200), (180, 30, 255))],
        "Grey": [((0, 0, 70), (180, 40, 200))],
    }

    max_pixels = 0
    detected_color = "Unknown"

    for color, ranges in color_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask += cv2.inRange(hsv, lower, upper)

        pixel_count = mask.sum()
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            detected_color = color

    confidence = min((max_pixels / (hsv.size)) * 100, 100)
    return detected_color, confidence


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Color"):
        color, confidence = detect_color_hsv(image)

        st.success(f"🎯 Predicted Color: **{color.upper()}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")



