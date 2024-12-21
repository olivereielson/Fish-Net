import geopandas as gpd
from shapely.geometry import Point
from PIL import Image, ImageDraw
import os
import pandas as pd
from shapely.wkt import loads

Image.MAX_IMAGE_PIXELS = None

df = pd.read_csv("all.csv")
df["LAT"] = abs(df["LAT"])
df["LON"] = abs(df["LON"])

# print(gdf)

# Example image title
w_pixel_to_degree = 0.016667 / 550
h_pixel_to_degree = 0.016667 / 730
w_start_degree = 72.283333
h_start_degree = 41.251667
print(w_start_degree, h_start_degree)


def add_dots(image_path, result):
    image = Image.open(image_path)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Example coordinates for dots

    # Example dot radius
    dot_radius = 10

    # Add dots to the image
    print(len(result))
    for coord in result.values:
        # print(coord)
        lat = coord[3]
        lon = coord[2]
        print(lat, lon)

        lat = ((w_start_degree - lat) * (550 / 0.016667)) % 5000
        lon = ((h_start_degree - lon) * (730 / 0.016667)) % 5000
        print(lat, lon)

        draw.ellipse(
            [lat - dot_radius, lon - dot_radius, lat + dot_radius,
             lon + dot_radius],
            fill='blue')

    with open("data.txt", 'w') as file:
        file.write(f"{image_path},{lat - 10},{lon - 10},{lat + 10},{lon + 10}\n")
        file.close()

    # Save or display the modified image
    modified_image_path = image_path
    # image.save(modified_image_path)
    image.show()


def procces_image(image_title):
    _, lon_str, lat_str = os.path.splitext(image_title)[0].split('_')
    lon, lat = abs(float(lon_str)), abs(float(lat_str))

    # print(lon, lat)
    result = df[(df['LAT'] < lat) & (df['LAT'] > (lat - (5000 * w_pixel_to_degree))) & (df['LON'] < lon) & (
            df['LON'] > (lon - (5000 * h_pixel_to_degree)))]

    if len(result) > 0:
        add_dots("grids/" + image_title, result)


for image in os.listdir('grids/'):
    procces_image(image)

# add_dots("/Users/olivereielson/Desktop/13209-11-1995 copy.jpg", df)
