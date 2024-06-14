# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:22:24 2024

@author: Dell
"""

import os
import glob
from pystackreg import StackReg
from skimage.io import imread
import numpy as np
from PIL import Image

def save_registered_images(registered_stack, output_folder, file_format):
    for i, img in enumerate(registered_stack):
        filename = f'registered_{i}.{file_format}'
        file_path = os.path.join(output_folder, filename)
        
        # Save as PNG
        Image.fromarray(img).save(file_path)
        print(f"PNG image saved successfully at: {file_path}")
        
        # Save as TIFF
        np.save(file_path[:-3] + 'tif', img)
        print(f"TIFF image saved successfully at: {file_path[:-3] + 'tif'}")

input_folder = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/NDp'
ndpi_files = sorted(glob.glob(os.path.join(input_folder, '*.ndpi')))
output_folder_start = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/NDp_coregistered_middle_to_start/level5'
output_folder_end = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/NDp_coregistered_middle_to_end/level5'
os.makedirs(output_folder_start, exist_ok=True)
os.makedirs(output_folder_end, exist_ok=True)

# Determine the middle index
middle_index = len(ndpi_files) // 2

# Load the images into two separate stacks: one for middle to start, one for middle to end
img_stack_start = [imread(f, plugin='tifffile') for f in ndpi_files[:middle_index+1][::-1]]  # Reverse to start from middle to first
img_stack_end = [imread(f, plugin='tifffile') for f in ndpi_files[middle_index:]]

# Convert lists to numpy arrays
img_stack_start = np.array(img_stack_start)
img_stack_end = np.array(img_stack_end)

# Initialize StackReg
sr = StackReg(StackReg.RIGID_BODY)

# Register the stacks
registered_start = sr.register_transform_stack(img_stack_start, reference='previous')[::-1]
registered_end = sr.register_transform_stack(img_stack_end, reference='previous')

# Save the registered images
save_registered_images(registered_start, output_folder_start, 'png')
save_registered_images(registered_start, output_folder_start, 'tif')
save_registered_images(registered_end, output_folder_end, 'png')
save_registered_images(registered_end, output_folder_end, 'tif')
