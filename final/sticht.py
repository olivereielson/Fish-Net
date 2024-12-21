from PIL import Image
import os


def stitch_images(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)
    # Sort the files based on their names (assuming the names contain coordinates)
    # files.sort()

    # Initialize an empty list to store the images
    images = []
    paths = []

    # Iterate through the files and open each image
    for file in files:
        file_path = os.path.join(directory, file)

        # Check if the file has a common image extension
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image = Image.open(file_path)
            images.append(image)
            paths.append(file_path)

    # Determine the dimensions of the stitched image
    total_width = 999 * 8
    total_height = 685 * 20

    # Create a new image with the calculated dimensions
    stitched_image = Image.new('RGB', (total_width, total_height))

    # Paste each individual image into the stitched image

    for image, path in zip(images, paths):
        _, _, x, y = path.split(".")[0].split("_")
        x = int(x)
        y = int(y)
        stitched_image.paste(image, (999 * x, 685 * y))
        image.close()

    # Save the final stitched image
    stitched_image.save('stitched_image.png')
    stitched_image.close()


# Specify the directory where individual screenshots are stored
screenshot_directory = "/Users/olivereielson/Desktop/stolen_goods"

# Call the function to stitch images
stitch_images(screenshot_directory)
