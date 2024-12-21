import datetime
import os

import keyboard
import time
import pyautogui
from PIL import ImageGrab
import math

### NOTE...I adapted this code from a website as it had nothing to do with the content of the class
# and I did not know how to scrape maps off the web myself. It had nothing to do with making the model
# and was just a way to download the data.

time.sleep(5)

screen_width, screen_height = pyautogui.size()

left, down = 0, 0

zoom = 16
pixel_width = 180
lat_width = abs(73.6248779296875 - 73.619384765625)
lon_width = abs(41.00477542222947 - 41.000629848685385)

start_lat = 71.48141
start_lon = 41.36098


# 147 to the top


def num2deg(xtile, ytile, zoom):
    n = 1 << zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def pixel_to_coord(px, py):
    lat = (px * 999) * (lat_width / pixel_width)
    lon = (py * 685) * (lon_width / pixel_width)
    lat = start_lat - lat
    lon = start_lon - lon
    return lat, lon


def delete_all_images(directory):
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        if any(file.lower().endswith(ext) for ext in image_extensions):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


def drag_right():
    start_x, start_y = screen_width - 50, 500  # You can adjust the starting point as needed

    # Set the distance to move the mouse for the drag
    drag_distance = -1050  # You can adjust the distance as needed

    # Move the mouse to the starting position
    pyautogui.moveTo(start_x, start_y)

    # Simulate a mouse button press (left button down)
    pyautogui.mouseDown()

    # Move the mouse to simulate dragging
    pyautogui.moveRel(drag_distance, 0, duration=1)  # Move to the right
    pyautogui.mouseUp()


def drag_left():
    start_x, start_y = 50, 500  # You can adjust the starting point as needed

    # Set the distance to move the mouse for the drag
    drag_distance = 1050  # You can adjust the distance as needed

    # Move the mouse to the starting position
    pyautogui.moveTo(start_x, start_y)

    # Simulate a mouse button press (left button down)
    pyautogui.mouseDown()

    # Move the mouse to simulate dragging
    pyautogui.moveRel(drag_distance, 0, duration=1)  # Move to the right
    pyautogui.mouseUp()


def drag_down():
    start_x, start_y = screen_width / 2, screen_height - 130  # You can adjust the starting point as needed
    drag_distance = -1 * (screen_height - 130 - 230)  # You can adjust the distance as needed
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    pyautogui.moveRel(0, drag_distance, duration=1)  # Move to the right
    pyautogui.mouseUp()


def screen_shot():
    time.sleep(1)
    pyautogui.moveTo(10, 10)

    folder_path = "/Users/olivereielson/Desktop/stolen_goods"

    dimensions = (50, 230, 1049, screen_height - 165)
    screenshot = ImageGrab.grab(bbox=dimensions)

    # Generate a timestamp for the file name

    lat, lon = pixel_to_coord(left, down)
    # Specify the file name
    file_name = f"stolen_{left}_{down}.png"

    # Specify the full path for saving the screenshot
    file_path = f"{folder_path}/{file_name}"

    # Save the screenshot
    screenshot.save(file_path)


delete_all_images("/Users/olivereielson/Desktop/stolen_goods")

for c in range(20):
    for r in range(7):
        print(left, down)
        screen_shot()
        if c % 2 == 0:
            drag_right()
            left += 1
        else:
            drag_left()
            left -= 1
        screen_shot()

    drag_down()
    down += 1
    screen_shot()
    print(left, down)
