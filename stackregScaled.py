# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:14:20 2024

@author: Dell
"""

import os
import glob
from openslide import open_slide
from pystackreg import StackReg
import numpy as np
from PIL import Image
from skimage.transform import resize

def save_image(image, filename, output_folder, file_format):
    image_pil = Image.fromarray(image).convert("L")
    file_path = os.path.join(output_folder, f"{filename}.{file_format}")
    image_pil.save(file_path)
    print(f"{file_format.upper()} image saved successfully at: {file_path}")

def rgba_to_gray(rgba_image):
    rgb_image = rgba_image[:, :, :3]
    gray_image = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140]) * rgba_image[..., 3] / 255.0
    return gray_image

def process_slide(slide_path, level, target_size=None):
    slide = open_slide(slide_path)
    dims = slide.level_dimensions
    level_img = slide.read_region((0, 0), level, dims[level])
    level_img_np = np.array(level_img, dtype='uint8')
    level_img_np_grayscale = rgba_to_gray(level_img_np)
    if target_size is not None:
        level_img_np_grayscale_resized = resize(level_img_np_grayscale, target_size, anti_aliasing=True)
        return level_img_np_grayscale_resized
    else:
        return level_img_np_grayscale

input_folder = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/NDp'
ndpi_files = sorted(glob.glob(os.path.join(input_folder, '*.ndpi')))
output_folder_start = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/Scaled/Stack_transform_middle_to_start/previous'
output_folder_end = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/Scaled/Stack_transform_middle_to_end/previous'
output_folder_start_first = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/Scaled/Stack_transform_middle_to_start/first'
output_folder_end_first = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/Scaled/Stack_transform_middle_to_end/first'
os.makedirs(output_folder_start, exist_ok=True)
os.makedirs(output_folder_end, exist_ok=True)
os.makedirs(output_folder_start_first, exist_ok=True)
os.makedirs(output_folder_end_first, exist_ok=True)

# Determine the middle index
if len(ndpi_files) % 2 == 0:
    middle_index = len(ndpi_files) // 2
else:
    middle_index = (len(ndpi_files) - 1) // 2

# Load the middle image and get its dimensions
middle_image_path = ndpi_files[middle_index]
middle_image = process_slide(middle_image_path, level=5)
target_height, target_width = middle_image.shape[:2]

# Load the images into two separate stacks: one for middle to start, one for middle to end
img_stack_start = [process_slide(f, level=5, target_size=(target_height, target_width)) for f in ndpi_files[:middle_index+1][::-1]]
img_stack_end = [process_slide(f, level=5, target_size=(target_height, target_width)) for f in ndpi_files[middle_index:]]

# Convert lists to numpy arrays
img_stack_start = np.array(img_stack_start)
img_stack_end = np.array(img_stack_end)

# Initialize StackReg
sr = StackReg(StackReg.SCALED_ROTATION)

# Register the stacks
registered_start = sr.register_transform_stack(img_stack_start, reference='previous')[::-1]
registered_end = sr.register_transform_stack(img_stack_end, reference='previous')

# Save the registered images
for i, registered_image in enumerate(registered_start):
    save_image(registered_image, f'registered_{i}', output_folder_start, 'png')
    save_image(registered_image, f'registered_{i}', output_folder_start, 'tif')

for i, registered_image in enumerate(registered_end):
    save_image(registered_image, f'registered_{i}', output_folder_end, 'png')
    save_image(registered_image, f'registered_{i}', output_folder_end, 'tif')

# Register the stacks to the first image in each stack
registered_start_first = sr.register_transform_stack(img_stack_start, reference='first')[::-1]
registered_end_first = sr.register_transform_stack(img_stack_end, reference='first')

# Save the registered images
for i, registered_image in enumerate(registered_start_first):
    save_image(registered_image, f'registered_{i}', output_folder_start_first, 'png')
    save_image(registered_image, f'registered_{i}', output_folder_start_first, 'tif')

for i, registered_image in enumerate(registered_end_first):
    save_image(registered_image, f'registered_{i}', output_folder_end_first, 'png')
    save_image(registered_image, f'registered_{i}', output_folder_end_first, 'tif')
