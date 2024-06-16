import os
import random
from skimage.feature import hog
from skimage import io, color, transform, exposure
import numpy as np
import matplotlib.pyplot as plt

# funkcija koja resize-a, crop-a sliku, računa i vizualizira hog značajke

def extract_hog_features(image, image_size=(250, 250), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True):
    if image.shape[2] == 4:
        image = color.rgba2rgb(image)
    # Resize <3
    resized_image = transform.resize(image, image_size, anti_aliasing=True)
    # Grayscale <3
    grayscale_image = color.rgb2gray(resized_image)
    # HOG značajke
    fd, hog_image = hog(grayscale_image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=visualize, block_norm='L2-Hys')
    return hog_image

test_folder_path = 'BigData/Train'

# lista svih slika u testnom folderu
image_files = [f for f in os.listdir(test_folder_path) if f.endswith('.png') or f.endswith('.jpg')]

# ako nema fajla
if not image_files:
    raise FileNotFoundError("No Train found in the test folder.")

# Nasumični odabir slike
random_image_filename = random.choice(image_files)
random_image_path = os.path.join(test_folder_path, random_image_filename)
print(f"Selected image: {random_image_filename}")

# učitaj sliku
image = io.imread(random_image_path)

# parametri - HOG
pixels_per_cell_values = [(2, 2), (4, 4), (8, 8), (16, 16)]
cells_per_block_values = [(1, 1), (2, 2)]

# prikaz
fig, axes = plt.subplots(len(pixels_per_cell_values), len(cells_per_block_values), figsize=(15, 15))
fig.suptitle('HOG Features with Different Parameters', fontsize=16)

# Loop over the parameter values and display the HOG Train
for i, pixels_per_cell in enumerate(pixels_per_cell_values):
    for j, cells_per_block in enumerate(cells_per_block_values):
        hog_image = extract_hog_features(image, image_size=(250, 250), pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        ax = axes[i, j]
        ax.imshow(hog_image, cmap='gray')
        ax.set_title(f'Pixels per cell: {pixels_per_cell}\nCells per block: {cells_per_block}')
        ax.axis('off')

# Display the table of Train :)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
