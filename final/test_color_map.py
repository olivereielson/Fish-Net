import cv2
import numpy as np


# Load the image


def remove_purple(image):
    upper_purple = np.array([204, 130, 204])
    lower_purple = np.array([155, 15, 155])

    mask = cv2.inRange(image, lower_purple, upper_purple)

    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Replace pixels in the original image
    image[dilated_mask > 0] = [255, 255, 255]

    return image


def remove_green(image):
    upper_green = np.array([223, 263, 237])
    lower_green = np.array([197, 237, 210])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.inRange(image, lower_green, upper_green)

    # Replace pixels in the original image
    image[mask > 0] = [255, 255, 255]

    return image


def remove_black(image):
    lower_green = np.array([0, 0, 0])
    upper_green = np.array([15, 15, 15])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.inRange(image, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Replace pixels in the original image
    image[dilated_mask > 0] = [255, 255, 255]

    return image


def remove_darker_than_and_compare(image, reference_average, threshold=(15, 15, 15), grid_size=100):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask for pixels darker than the threshold
    dark_mask = np.any(image < threshold, axis=-1)

    # Iterate through each dark pixel
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            if dark_mask[y, x]:  # Dark pixel found
                # Calculate the average color of the 100x100 grid around the pixel
                grid = image[max(0, y - grid_size):min(image.shape[0], y + grid_size),
                       max(0, x - grid_size):min(image.shape[1], x + grid_size)]
                average_color = np.mean(grid, axis=(0, 1)).astype(int)

                # Compare with the reference average color
                if np.all(average_color <= reference_average):
                    # Turn the specific dark pixel to white
                    image[y, x] = [255, 0, 0]

    return image



image2 = cv2.imread("/Users/olivereielson/Desktop/test.jpg")
average_color = np.mean(image2, axis=(0, 1)).astype(int)

image = cv2.imread('/Users/olivereielson/Desktop/13215-1.png')

image = remove_purple(image)
# image = remove_green(image)
image = remove_darker_than_and_compare(image, average_color)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.imshow("no color", image, )
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output_image.jpg', image)
