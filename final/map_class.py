import operator
import os
import sys

from PIL import Image, ImageDraw
from mpmath import mp


class Map_Location:

    def __init__(self, w_start_degree, h_start_degree, degree_w, pixel_w, degree_h, pixel_h,
                 map_path, location_name, square_tiles=True,stolen=False):

        self.w_pixel_to_degree = (degree_w / pixel_w)
        self.h_pixel_to_degree = (degree_h / pixel_h)
        self.w_degree_to_pixel = (pixel_w / degree_w)
        self.h_degree_to_pixel = (pixel_h / degree_h)
        self.w_start_degree = w_start_degree
        self.h_start_degree = h_start_degree

        self.map_path = map_path
        self.exclusion = []
        self.location_name = location_name
        self.square_tiles = square_tiles

        self.is_stolen=stolen

    def add_exclusion(self, box):
        self.exclusion.append(box)

    def get_path(self):
        return self.map_path

    def get_boats(self, lat, lon, raw_data, tile_size):

        result = raw_data

        result = result[(result['LAT'] < lat)]
        result = result[result['LAT'] > (lat - (tile_size * self.w_pixel_to_degree))]
        result = result[result['LON'] < lon]
        result = result[result['LON'] > (lon - (tile_size * self.h_pixel_to_degree))]

        if self.exclusion is None:
            return result

        for box in self.exclusion:
            xmin = min(box[1], box[3])
            xmax = max(box[1], box[3])

            ymin = min(box[0], box[2])
            ymax = max(box[0], box[2])

            result = result[
                ~(((result["LAT"] > xmin) & (result["LAT"] < xmax)) &
                  ((result["LON"] > ymin) & (result["LON"] < ymax)))
            ]

        return result

    def cord_to_pixel(self, lat, lon, tile_size):
        if tile_size == 0:
            lat = int(((self.w_start_degree - lat) * self.w_degree_to_pixel))
            lon = int(((self.h_start_degree - lon) * self.h_degree_to_pixel))
            return lat, lon
        else:
            lat = int(((self.w_start_degree - lat) * self.w_degree_to_pixel) % tile_size)
            lon = int(((self.h_start_degree - lon) * self.h_degree_to_pixel) % tile_size)
            return lat, lon

    def pixel_to_cord(self, x, y, lat, lon):
        lat = (lat - (x * self.w_pixel_to_degree))
        lon = (lon - (y * self.h_pixel_to_degree))
        return lat, lon

    def draw_exclusion(self):
        print("Drawing exclusion zone for " + self.location_name)
        image = Image.open(self.map_path)
        draw = ImageDraw.Draw(image)
        for coord in self.exclusion:
            x1, y1 = self.cord_to_pixel(coord[1], coord[0], 0)
            x2, y2 = self.cord_to_pixel(coord[3], coord[2], 0)

            draw = ImageDraw.Draw(image)

            rectangle_coords = (int(x1), int(y1), int(x2), int(y2))

            draw.rectangle(rectangle_coords, fill=None, outline="red", width=50)
        image.show()


    def gen_tiles(self, tile_size, file_path):
        Image.MAX_IMAGE_PIXELS = None

        assert (os.path.exists(self.map_path))

        with Image.open(self.map_path) as img:
            img_width, img_height = img.size
            print(img_width, img_height)

            for x in range(0, img_width, tile_size):
                for y in range(0, img_height, tile_size):
                    left = x
                    upper = y
                    right = min(x + tile_size, img_width)
                    lower = min(y + tile_size, img_height)

                    if (right - left != tile_size or lower - upper != tile_size) and self.square_tiles:
                        continue

                    sub_image = img.crop((left, upper, right, lower))
                    x_name = (self.w_start_degree - (x * self.w_pixel_to_degree))
                    y_name = (self.h_start_degree - (y * self.h_pixel_to_degree))

                    filename = f"{self.location_name}_{x_name}_{y_name}.png"
                    sub_image.save(os.path.join(file_path, filename))
