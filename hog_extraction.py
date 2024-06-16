# hog_extraction.py
import os
from skimage.feature import hog
from skimage import io, color, transform
import numpy as np
import joblib

# Path do foldera
folder_path = 'BigData/Train'

# Parameteri -  HOG
orientations = 8
pixels_per_cell = (2, 2)
cells_per_block = (1, 1)
image_size = (250, 250)  # Size to resize Train after cropping


# crop, resize i ekstakcija HOG značajki
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
    fd = hog(resized_image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
             visualize=False, block_norm='L2-Hys')
    return fd


# Liste u kojima smještamo HOG značajke i labele
# sa 1 je označena slika koja ima labelu 'limit', 0 u protivnom
hog_features = []
labels = []

# iteriramo kroz sve fajlove u folderu
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(folder_path, filename)
        fd = extract_hog_features(image_path)
        hog_features.append(fd)
        labels.append(1 if 'limit' in filename else 0)

X = np.array(hog_features)
y = np.array(labels)

# spasi HOG značajke i labele
# ovo koristimo za treniranje SVM
data_path = 'hog_features_labels.joblib'
joblib.dump((X, y), data_path)
print(f'HOG features and labels saved to {data_path}')
