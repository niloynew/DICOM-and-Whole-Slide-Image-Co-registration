
"""
Created on Wed Feb 28 23:46:20 2024

@author: Niloy Roy
"""
from pystackreg import StackReg
from openslide import open_slide
import openslide
import numpy as np
from matplotlib import pyplot as plt
#from skimage.color import rgb2gray
from skimage.transform import resize

def process_slide(slide_path, level):
    # Load the slide file into an object.
    slide = open_slide(slide_path)

    # Get slide properties
    slide_props = slide.properties
    print("Properties:", slide_props)

    print("Vendor is:", slide_props['openslide.vendor'])
    print("Pixel size of X in um is:", slide_props['openslide.mpp-x'])
    print("Pixel size of Y in um is:", slide_props['openslide.mpp-y'])

    # Objective used to capture the image
    objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    print("The objective power is: ", objective)

    # Get slide dimensions for the level 0 - max resolution level
    slide_dims = slide.dimensions
    print("Slide dimensions:", slide_dims)

    # Get a thumbnail of the image and visualize
    slide_thumb_600 = slide.get_thumbnail(size=(600, 600))
    slide_thumb_600.show()

    # Convert thumbnail to numpy array
    slide_thumb_600_np = np.array(slide_thumb_600)
    plt.figure(figsize=(8, 8))
    plt.imshow(slide_thumb_600_np)

    # Get slide dims at each level.
    dims = slide.level_dimensions
    num_levels = len(dims)
    print("Number of levels in this image are:", num_levels)
    print("Dimensions of various levels in this image are:", dims)

    # By how much are levels downsampled from the original image?
    factors = slide.level_downsamples
    print("Each level is downsampled by an amount of: ", factors)

    # Copy an image from a level
    level_dim = dims[level]
    level_img = slide.read_region((0, 0), level, level_dim)  # Pillow object, mode=RGBA

    # Convert the image to RGB
    level_img_RGB = level_img.convert('RGB')
    level_img_RGB.show()

    # Convert the image into a numpy array for processing
    level_img_np = np.array(level_img, dtype='uint8')
    

    return level_img_np

def rgba_to_gray(rgba_image):
    # Extract the RGB channels
    rgb_image = rgba_image[:, :, :3]
    # Convert to grayscale while considering alpha channel
    gray_image = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140]) * rgba_image[..., 3] / 255.0
    return gray_image

# Example usage:
slide_path = "F:/Germany_2022/TU Illmenau/hiwi/DataSets/NDp/FLM-005_J-13-2235_1564_HE.ndpi"
ref_img = process_slide(slide_path, level=9)

slide_path2 = "F:/Germany_2022/TU Illmenau/hiwi/DataSets/NDp/FLM-005_J-13-2235_1568_HE.ndpi"
offset_img = process_slide(slide_path2, level=8)



ref_gray = rgba_to_gray(ref_img)
offset_gray = rgba_to_gray(offset_img)
# Resize images to have the same dimensions
max_height = max(ref_gray.shape[0], offset_gray.shape[0])
max_width = max(ref_gray.shape[1], offset_gray.shape[1])
ref_resized = resize(ref_gray, (max_height, max_width))
offset_resized = resize(offset_gray, (max_height, max_width))






#Rigid Body transformation
sr = StackReg(StackReg.RIGID_BODY)
out_rot = sr.register_transform(ref_resized, offset_resized)

#Scaled Rotation transformation
#sr = StackReg(StackReg.SCALED_ROTATION)
#out_sca = sr.register_transform(ref_img, offset_img)


fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(ref_img, cmap='gray')
ax1.title.set_text('Input Image')

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(out_rot, cmap='gray')
ax3.title.set_text('Rigid Body')

plt.show()


from PIL import Image

# Convert the reference and registered image arrays to PIL Image
ref_image_pil = Image.fromarray(ref_img).convert("L")
registered_image_pil = Image.fromarray(out_rot).convert("L")


# Specify the file paths to save the images
ref_image_path = "F:/Germany_2022/TU Illmenau/hiwi/DataSets/NDp/co-registered/reference_image.png"
registered_image_path = "F:/Germany_2022/TU Illmenau/hiwi/DataSets/NDp/co-registered/registered_image.png"

# Save the reference image
ref_image_pil.save(ref_image_path)
print("Reference image saved successfully at:", ref_image_path)

# Save the registered image
registered_image_pil.save(registered_image_path)
print("Registered image saved successfully at:", registered_image_path)





