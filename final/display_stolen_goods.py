import os
import time

from PIL import ImageDraw
from PIL import Image
from shapely.wkt import loads
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# w_tile_size = 1000
# h_tile_size = 685
# lat_width = abs(73.6248779296875 - 73.619384765625)
# lon_width = abs(41.00477542222947 - 41.000629848685385)
# pixel_width = 180
#
# # w_pixel_to_degree = (lat_width / pixel_width)
# # h_pixel_to_degree = (lon_width / pixel_width)
#
# # start_lat = 73.10319
# start_lat = 73.10319
# start_lon = 41.15209
#
# count = 0


class Stolen_Goods:
    def __init__(self, start_lat, start_lon, w_tile_size, h_tile_size, lat_width, lon_width, pixel_width, pixel_height,
                 unwanted_mmsi=None, exclusion=None, image_file_paths="/Users/olivereielson/Desktop/stolen_goods",
                 raw_data="all.csv"):

        self.start_lat = start_lat
        self.start_lon = start_lon
        self.w_tile_size = w_tile_size
        self.h_tile_size = h_tile_size
        self.lat_width = lat_width
        self.lon_width = lon_width
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.unwanted_mmsi = unwanted_mmsi
        if exclusion is None:
            exclusion = []
        self.exclusion = exclusion
        self.image_file_paths = image_file_paths
        self.raw_data = pd.read_csv(raw_data)
        self.w_pixel_to_degree = (lat_width / pixel_width)
        self.h_pixel_to_degree = (lon_width / pixel_width)

        self.clean_data()

    def add_exclusion(self, box):
        self.exclusion.append(box)

    def clean_data(self):

        if self.unwanted_mmsi is not None:
            for mmsi in self.unwanted_mmsi:
                self.raw_data = self.raw_data[self.raw_data['MMSI'] != mmsi]

        # I was an idiot an mixed up lat and lon so this fixes them
        self.raw_data['geometry'] = self.raw_data['geometry'].apply(loads)
        self.raw_data['LAT'] = self.raw_data['geometry'].apply(lambda point: point.x)
        self.raw_data['LON'] = self.raw_data['geometry'].apply(lambda point: point.y)

        # remove neg cordinartes for simplicty
        self.raw_data["LAT"] = abs(self.raw_data["LAT"])
        self.raw_data["LON"] = abs(self.raw_data["LON"])

    def get_boats2(self, lat, lon):
        results = self.raw_data
        results = results[(results['LAT'] < lat)]
        results = results[results['LAT'] > (lat - (self.w_tile_size * self.w_pixel_to_degree))]
        results = results[results['LON'] < lon]
        results = results[results['LON'] > (lon - (self.h_tile_size * self.h_pixel_to_degree))]

        if self.exclusion is not None:

            for box in self.exclusion:
                xmin = min(box[1], box[3])
                xmax = max(box[1], box[3])

                ymin = min(box[0], box[2])
                ymax = max(box[0], box[2])

                # this is the bitwise operatotrs
                results = results[
                    ~(((results["LAT"] > xmin) & (results["LAT"] < xmax)) &
                      ((results["LON"] > ymin) & (results["LON"] < ymax)))
                ]

        return results

    def cord_to_pixel(self, lat, lon, lat_image, lon_image):

        lat = int((lat_image - lat) * (self.pixel_width / self.lat_width) % self.w_tile_size)
        lon = int(((lon_image - lon) * (self.pixel_width / self.lon_width)) % self.h_tile_size)
        return lat, lon

    def generate(self):

        file = open("data_labels.csv", 'w')
        file.write("image_path,label,x1,y1,x2,y2\n")

        for image_path in os.listdir(self.image_file_paths):
            if not os.path.splitext(image_path)[1] == ".png":
                continue

            if image_path == "stolen_72.27326645507813_41.16955894784169.png":
                continue

            name, lat_str, lon_str = os.path.splitext(image_path)[0].split('_')
            lon_image, lat_image = abs(float(lon_str)), abs(float(lat_str))

            result = self.get_boats2(lat_image, lon_image)
            for coord in result.values:
                lat = coord[2]
                lon = coord[3]

                p_x, p_y = self.cord_to_pixel(lat, lon, lat_image, lon_image)

                file.write(
                    f"grids/{image_path},1,{p_x - 25},{p_y - 25},{p_x + 25},{p_y + 25}\n")

        file.close()

    def show_images(self):
        for image_path in os.listdir(self.image_file_paths):
            df = pd.read_csv("data_labels.csv")
            image_df = df[df["image_path"] == "grids/" + image_path]
            print(image_path)
            if len(image_df) == 0:
                continue

            image = Image.open(self.image_file_paths + "/" + image_path)
            img_width, img_height = image.size

            fig, ax = plt.subplots()
            ax.imshow(image)

            for coord in image_df.values:
                x1 = int(coord[2])
                y1 = int(coord[3])
                x2 = min(int(coord[4]), img_width)
                y2 = min(int(coord[5]), img_height)

                rectangle_coords = (x1, y1, x2 - x1, y2 - y1)
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

            # Set custom title for the image window
            plt.title(image_path)
            plt.show()
            time.sleep(5)
