from PIL import Image
import os
import math


def delete_all_images_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def convert_to_dms(degrees):
    deg = int(degrees)
    minutes = abs((degrees - deg) * 60)
    seconds = (minutes - int(minutes)) * 60
    return f"{deg:02d}{int(minutes):02d}"

# Constants
w_pixel_to_degree = 0.01 / 550
w_start_degree = 72.17

h_pixel_to_degree = 0.01 / 730
h_start_degree = 41.10 + (3980 * h_pixel_to_degree)

height = 5000
width = 5000

input_image_path = "/Users/olivereielson/Desktop/13209-11-1995 copy.jpg"
output_folder = "grids"

Image.MAX_IMAGE_PIXELS = None

delete_all_images_in_folder(output_folder)

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Open the input image using Pillow
with Image.open(input_image_path) as img:
    # Get the size of the input image
    img_width, img_height = img.size

    # Loop through the image and split it into 150x150 sub-images
    for x in range(0, img_width, width):
        for y in range(0, img_height, height):
            left = x
            upper = y
            right = min(x + width, img_width)
            lower = min(y + height, img_height)

            # Crop the sub-image
            sub_image = img.crop((left, upper, right, lower))

            # Create a filename for the sub-image based on its position
            x_degrees = w_start_degree - (x * w_pixel_to_degree)
            y_degrees = h_start_degree - (y * h_pixel_to_degree)

            x_degrees_abs = abs(x_degrees)
            y_degrees_abs = abs(y_degrees)

            x_degrees_int = int(x_degrees_abs)
            y_degrees_int = int(y_degrees_abs)

            x_minutes = (x_degrees_abs - x_degrees_int) * 60
            y_minutes = (y_degrees_abs - y_degrees_int) * 60

            filename = f"{x_degrees_int:02d}.{convert_to_dms(x_minutes)}_{y_degrees_int:02d}.{convert_to_dms(y_minutes)}.png"

            # Save the sub-image to the output folder
            sub_image.save(os.path.join(output_folder, filename))

            print(f"Processed: {filename}")

print("Processing complete.")
