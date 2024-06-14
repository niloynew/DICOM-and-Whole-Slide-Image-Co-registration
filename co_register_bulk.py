# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:44:23 2024

@author: Niloy Roy
"""
from openslide import open_slide
import glob
import os
import tifffile
from pystackreg import StackReg
from skimage.transform import resize
#from skimage.color import rgba2rgb, rgb2gray
#from skimage.io import imsave
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
    #num_levels = len(dims)
	
	
	# Get the index of the highest level
    #highest_level_index = num_levels - 1

    # Get dimensions of the highest level
    #highest_level_dim = dims[highest_level_index]
	

    level_img = slide.read_region((0, 0), 5, dims[5])  # Pillow object, mode=RGBA

   

    # Convert the image into a numpy array for processing
    level_img_np = np.array(level_img, dtype='uint8')
    
    level_img_np_grayscale = rgba_to_gray(level_img_np)

    return level_img_np_grayscale

# Function to process and resize images
# order 4 = Bi-quartic interpolation
def process_and_resize_image(filename, target_height, target_width):
    img = process_slide(filename)
    # Assuming process_slide returns a numpy array
    img_resized = resize(img, (target_height, target_width), order=4, anti_aliasing=True, preserve_range=True).astype(np.uint8)
    
    return img_resized

# Load the NDPI files
# Set the desired input folder path
input_folder = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/NDp'

# Get the list of NDPI files in the input folder
ndpi_files = glob.glob(os.path.join(input_folder, '*.ndpi'))

if len(ndpi_files) % 2 == 0:
    middle_index = len(ndpi_files) // 2
else:
    middle_index = (len(ndpi_files) - 1) // 2

# Initialize StackReg
sr = StackReg(StackReg.RIGID_BODY)

# Process and resize images starting from the middle index
reference_image_path = ndpi_files[middle_index]
reference_image = process_slide(reference_image_path)
reference_height, reference_width = reference_image.shape[:2]

# Set the desired folder pathpair_by_pair_Previous
output_folder = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/Result/Middle_To_End/pairbypair_withPrevious'
output_folder_1 = 'F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/Result/Middle_To_Start/pair_by_pair_Previous'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_1, exist_ok=True)


# Save the reference image
tifffile.imwrite(os.path.join(output_folder_1, f'reference_{middle_index}.tif'), reference_image.astype(np.uint8))
save_image(reference_image, f'reference_{middle_index}', output_folder_1, 'png')


for i in range(middle_index - 1, -1, -1):
    # Resize the offset image to match the reference image dimensions
    offset_image_resized = process_and_resize_image(ndpi_files[i], reference_height, reference_width)
    
    # Co-register the offset image to the reference image
    registered_image = sr.register_transform(reference_image, offset_image_resized)
    
    # Save the reference and registered image pair
    #registered_image_pil = Image.fromarray(registered_image).convert("L")
    #registered_image_pil.save(output_folder)
    
 	
    tifffile.imwrite(os.path.join(output_folder_1, f'registered_{i}.tif'), registered_image.astype(np.uint8))
    save_image(registered_image, f'registered_{i}', output_folder_1, 'png')
    
    # Update the reference image and its dimensions for the next iteration
    reference_image = registered_image
    reference_height, reference_width = reference_image.shape[:2] 


for i in range(middle_index + 1, len(ndpi_files)):
    # Resize the offset image to match the reference image dimensions
    offset_image_resized = process_and_resize_image(ndpi_files[i], reference_height, reference_width)
    
    # Co-register the offset image to the reference image
    registered_image = sr.register_transform(reference_image, offset_image_resized)
    
    # Save the reference and registered image pair
    #registered_image_pil = Image.fromarray(registered_image).convert("L")
    #registered_image_pil.save(output_folder)
    
 	
    tifffile.imwrite(os.path.join(output_folder, f'registered_{i}.tif'), registered_image.astype(np.uint8))
    #save_image(registered_image, f'registered_{i}', output_folder, 'png')
    
    # Update the reference image and its dimensions for the next iteration
    reference_image = registered_image
    reference_height, reference_width = reference_image.shape[:2]
    
    

    
    
    
    
    
    
    
    
    
    
    