import os
import random
from skimage.feature import hog
from skimage import io, color, transform
import numpy as np
import joblib
import matplotlib.pyplot as plt

""" Generira se random slika iz dijela dataseta namijenjenog za testiranje 
        Pravi se predikcija da li je slika zaista znak ograničenja brzine!"""

# Load SVM modela
model_path = '../svm_model.joblib'
svm = joblib.load(model_path)

# HOG parametri (treba da odgovaraju onima iz training dijela)
orientations = 8
pixels_per_cell = (2, 2)
cells_per_block = (1, 1)
image_size = (250, 250)


def extract_hog_features(image_path, crop_size=(250, 250), resize_size=(250, 250)):
    image = io.imread(image_path)
    if image.shape[2] == 4:
        image = color.rgba2rgb(image)
    # Crop the image to the center
    start_x = (image.shape[1] - crop_size[0]) // 2
    start_y = (image.shape[0] - crop_size[1]) // 2
    cropped_image = image[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]
    resized_image = transform.resize(cropped_image, resize_size, anti_aliasing=True)
    if resized_image.ndim > 2:
        resized_image = color.rgb2gray(resized_image)
    fd = hog(resized_image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=False, block_norm='L2-Hys')
    return fd

test_folder_path = 'BigData/Test'

# lista svih slika u test folderu
image_files = [f for f in os.listdir(test_folder_path) if f.endswith('.png') or f.endswith('.jpg')]

# da li je slučajno ostao train dio
if not image_files:
    raise FileNotFoundError("No Train found in the test folder.")

# odabir random slike
random_image_filename = random.choice(image_files)
random_image_path = os.path.join(test_folder_path, random_image_filename)
print(f"Selected image: {random_image_filename}")

# hog značajke za random sliku
random_image_hog_features = extract_hog_features(random_image_path).reshape(1, -1)

# Predikicja <3
prediction = svm.predict(random_image_hog_features)
prediction_text = "Znak ograničenja!" if prediction[0] == 1 else "Nije znak ograničenja!"
print(f'Prediction: {prediction_text}')

# Prikaz <3
image = io.imread(random_image_path)
plt.imshow(image)
plt.title(f'Prediction: {prediction_text}')
plt.axis('off')
plt.show()
