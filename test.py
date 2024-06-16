import os
from skimage.feature import hog
from skimage import io, color, transform
import numpy as np
import joblib
import matplotlib.pyplot as plt

""" prkaz do 6 slika koje nisu znak brzine"""

# loadamo svm model
model_path = '../svm_model.joblib'
svm = joblib.load(model_path)

# HOG parametri (isti kao i za trening)
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

# Lista svih slika u folderu
image_files = [f for f in os.listdir(test_folder_path) if f.endswith('.png') or f.endswith('.jpg')]

if not image_files:
    raise FileNotFoundError("No Train found in the test folder.")

# Lista slika - predikacija da nisu speedsigns
speedlimit_sign_images = []

# iteriramo kroz svaku sliku u fajlu
for filename in image_files:
    image_path = os.path.join(test_folder_path, filename)
    fd = extract_hog_features(image_path)
    # reshaping deskriptora
    fd_reshaped = fd.reshape(1, -1)
    # predikcija <3
    prediction = svm.predict(fd_reshaped)
    if prediction[0] == 0:  # 1 indicates not a sign
        speedlimit_sign_images.append(image_path)

# Prikaz rezultata
if speedlimit_sign_images:
    plt.figure(figsize=(15, 10))
    plt.suptitle('Nisu znakovi ograničenja brzine', fontsize=16)
    num_images = len(speedlimit_sign_images)
    columns = 3
    rows = (num_images // columns) + 1
    for i, image_path in enumerate(speedlimit_sign_images):
        if i > 6:
            break
        image = io.imread(image_path)
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Image {i + 1}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
else:
    print("Nema detektovanih znkova.")
