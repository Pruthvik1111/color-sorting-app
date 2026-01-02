import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, filedialog

# Load trained model
model = tf.keras.models.load_model("sorter_model.h5")

IMG_SIZE = 128

# ⚠️ MUST match train_data.class_indices order
class_names = ['blue', 'green', 'red']

def predict_color(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Could not read the selected image.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    color = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print("\n✅ Prediction Result")
    print(f"Predicted Color : {color}")
    print(f"Confidence      : {confidence:.2f}%")

# 📂 File picker
def select_image():
    root = Tk()
    root.withdraw()  # hide main window

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        predict_color(file_path)
    else:
        print("❌ No image selected.")

if __name__ == "__main__":
    select_image()
