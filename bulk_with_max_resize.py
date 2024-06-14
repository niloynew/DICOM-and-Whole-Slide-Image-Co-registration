# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:44:23 2024

@author: Niloy Roy
"""
from openslide import open_slide
import glob
import os
from pystackreg import StackReg
from skimage.transform import resize
import numpy as np
from PIL import Image

def save_image(image, filename, output_folder, file_format):
    # Convert the image to a PIL Image object
    image_pil = Image.fromarray(image).convert("L")
    
    # Specify the file path to save the image
    file_path = os.path.join(output_folder, f"{filename}.{file_format}")
    
    # Save the image
    image_pil.save(file_path)
    
    print(f"{file_format.upper()} image saved successfully at: {file_path}")

def rgba_to_gray(rgba_image):
    # Extract the RGB channels
    rgb_image = rgba_image[:, :, :3]
    # Convert to grayscale while considering alpha channel
    gray_image = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140]) * rgba_image[..., 3] / 255.0
    return gray_image

def process_slide(slide_path):
    # Load the slide file into an object.
    slide = open_slide(slide_path)

    # Get slide dims at each level.
    dims = slide.level_dimensions
    num_levels = len(dims)

    # Get the index of the highest level
    highest_level_index = num_levels - 1

    # Get dimensions of the highest level
    highest_level_dim = dims[highest_level_index]

    level_img = slide.read_region((0, 0), highest_level_index, highest_level_dim)  # Pillow object, mode=RGBA

    # Convert the image into a numpy array for processing
    level_img_np = np.array(level_img, dtype='uint8')
    
    level_img_np_grayscale = rgba_to_gray(level_img_np)

    return level_img_np_grayscale

# Function to process and resize images
def resize_image(img, max_height, max_width):
    #img = process_slide(filename)
    # Assuming process_slide returns a numpy array
    img_resized = resize(img, (max_height, max_width), order=4, anti_aliasing=True, preserve_range=True).astype(np.uint8)
    return img_resized

# Load the NDPI files
# Set the desired input folder path
input_folder = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/NDp'

#'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/NDp'
#'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/test_input'

# Get the list of NDPI files in the input folder
ndpi_files = glob.glob(os.path.join(input_folder, '*.ndpi'))

if len(ndpi_files) % 2 == 0:
    middle_index = len(ndpi_files) // 2
else:
    middle_index = (len(ndpi_files) - 1) // 2

# Initialize StackReg
sr = StackReg(StackReg.RIGID_BODY)

# Set the desired folder path
output_folder = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/NDp_coregistered'
#'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/test_output'
#'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/NDp_coregistered'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process and resize images starting from the middle index
reference_image_path = ndpi_files[middle_index]
reference_image = process_slide(reference_image_path)
reference_height, reference_width = reference_image.shape[:2]

# Save the reference image
save_image(reference_image, f'reference_{middle_index}', output_folder, 'png')

for i in range(middle_index + 1, len(ndpi_files)):
    # Process the offset image
    offset_image_path = ndpi_files[i]
    offset_image = process_slide(offset_image_path)
    offset_height, offset_width = offset_image.shape[:2]

    # Find the maximum height and width of the reference and offset images
    max_height = max(reference_height, offset_height)
    max_width = max(reference_width, offset_width)

    # Resize the reference and offset images to the maximum dimensions
    reference_image_resized = resize_image(reference_image, max_height, max_width)
    offset_image_resized = resize_image(offset_image, max_height, max_width)

    # Co-register the offset image to the reference image
    registered_image = sr.register_transform(reference_image_resized, offset_image_resized)

    # Save the reference and registered images
    save_image(reference_image_resized, f'reference_{middle_index}', output_folder, 'png')
    save_image(registered_image, f'registered_{i}', output_folder, 'png')

    # Update the reference image and its dimensions for the next iteration
    reference_image = registered_image
    reference_height, reference_width = reference_image.shape[:2]