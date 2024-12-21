import warnings
from tkinter import Image

import cv2
import numpy as np
from shapely.geometry import Point

import torch
from PIL import Image, ImageDraw
import os
import pandas as pd
from PIL.Image import DecompressionBombError
from mpmath import mp  # Added import statement
from torchvision import transforms
from torchvision.io import read_image

from final.map_class import Map_Location
from typing import List
from shapely.wkt import loads

Image.MAX_IMAGE_PIXELS = None


class DataSet:
    def __init__(self, tile_size, ais_file_path, image_file_paths, locations, unwanted_mmsi=None,
                 output_path="data_labels.csv", prediction_path="out.csv", regenerate=False
                 ):
        # data files
        self.output_path = output_path
        self.ais_file_path = ais_file_path
        self.image_file_paths = image_file_paths
        self.prediction_path = prediction_path

        self.regenerate = regenerate

        # locations

        self.locations: List[Map_Location] = locations
        assert len(locations) != 0

        # Map Data
        # self.w_pixel_to_degree = 0.016667 / 550
        # self.h_pixel_to_degree = 0.016667 / 730
        #
        # self.w_degree_to_pixel = 550 / 0.016667
        # self.h_degree_to_pixel = 730 / 0.016667
        #
        # self.w_start_degree = 72.283333
        # self.h_start_degree = 41.256567

        self.tile_size = tile_size

        # unwanted boats
        self.unwanted_mmsi = unwanted_mmsi
        # unparsed Data
        self.raw_data = pd.read_csv(self.ais_file_path)
        # set acccuracy
        mp.dps = 50

    def clean_data(self):
        for mmsi in self.unwanted_mmsi:
            self.raw_data = self.raw_data[self.raw_data['MMSI'] != mmsi]

        # I was an idiot an mixed up lat and lon so this fixes them
        self.raw_data['geometry'] = self.raw_data['geometry'].apply(loads)
        self.raw_data['LAT'] = self.raw_data['geometry'].apply(lambda point: point.x)
        self.raw_data['LON'] = self.raw_data['geometry'].apply(lambda point: point.y)

        # remove neg cordinartes for simplicty
        self.raw_data["LAT"] = abs(self.raw_data["LAT"])
        self.raw_data["LON"] = abs(self.raw_data["LON"])

    # def remove_purple(self, image_path):
    #     image = cv2.imread(image_path)
    #     upper_purple = np.array([204, 130, 204])
    #     lower_purple = np.array([105, 68, 108])
    #
    #     mask = cv2.inRange(image, lower_purple, upper_purple)
    #
    #     kernel = np.ones((5, 5), np.uint8)
    #     dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    #
    #     # Replace pixels in the original image
    #     image[dilated_mask > 0] = [255, 255, 255]
    #     cv2.imwrite(image_path, image)

    def procces_image(self, path, file, location):
        name, lat_str, lon_str = os.path.splitext(path)[0].split('_')

        if name != self.image_file_paths + location.location_name:
            return

        lon, lat = abs(float(lon_str)), abs(float(lat_str))

        result = location.get_boats(lat, lon, self.raw_data, self.tile_size)
        for coord in result.values:
            lat = coord[2]
            lon = coord[3]

            p_x, p_y = location.cord_to_pixel(lat, lon, self.tile_size)

            if p_x + 10 < self.tile_size and p_y + 10 < self.tile_size and p_x - 10 > 0 and p_y - 10 > 0:
                file.write(
                    f"{path},1,{p_x - 30},{p_y - 30},{p_x + 30},{p_y + 30}\n")

    def delete_all_images_in_folder(self):
        for filename in os.listdir(self.image_file_paths):
            file_path = os.path.join(self.image_file_paths, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    def current_image_size(self):

        if len(os.listdir(self.image_file_paths)) == 0:
            return False

        image_path = os.listdir(self.image_file_paths)[0]
        with Image.open(self.image_file_paths + image_path) as img:
            img_width, img_height = img.size

        if not (img_width == self.tile_size and img_height == self.tile_size):
            return False

        path_list = os.listdir(self.image_file_paths)
        for locations in self.locations:
            if not any(locations.location_name in path_name for path_name in path_list):
                return False

        return True

    def generate(self):

        if not self.current_image_size() or self.regenerate:

            print("Generating tiles")
            self.delete_all_images_in_folder()
            os.makedirs(self.image_file_paths, exist_ok=True)

            for location in self.locations:
                location.gen_tiles(self.tile_size, self.image_file_paths)

        self.clean_data()
        file = open(self.output_path, 'w')
        file.write("image_path,label,x1,y1,x2,y2\n")

        for location in self.locations:
            for image_path in os.listdir(self.image_file_paths):
                self.procces_image(self.image_file_paths + image_path, file, location)

        file.close()

    def draw_image(self, path):
        df = pd.read_csv(self.output_path)
        image_df = df[df["image_path"] == self.image_file_paths + path]
        if len(image_df) == 0:
            return
        image = Image.open(self.image_file_paths + path)
        img_width, img_height = image.size

        draw = ImageDraw.Draw(image)

        for coord in image_df.values:
            x1 = int(coord[2])
            y1 = int(coord[3])
            x2 = min(int(coord[4]), img_width)
            y2 = min(int(coord[5]), img_height)

            draw = ImageDraw.Draw(image)

            rectangle_coords = (int(x1), int(y1), int(x2), int(y2))

            draw.rectangle(rectangle_coords, fill=None, outline="red", width=3)
        image.show()

    def draw_exclusion(self):
        for location in self.locations:
            location.draw_exclusion()

    def get_location(self, name):
        for location in self.locations:
            if location.location_name == name:
                return location
        raise Exception("Location not found")

    def show_images(self):
        for image_path in os.listdir(self.image_file_paths):
            self.draw_image(image_path)

    def get_predictions(self, image, model, lat, lon, location, confidence_threshold=0.25):

        lat, lon = location.cord_to_pixel(lat, lon, )
        out_csv = open(self.prediction_path, 'a')
        model.eval()
        predictions = model([image])
        for prediction in predictions:
            boxes = prediction["boxes"].tolist()
            scores = prediction["scores"].tolist()

            boxes = [box for box, score in zip(boxes, scores) if score > confidence_threshold]
            for box, score in zip(boxes, scores):
                x1 = (box[0] * self.tile_size) + lat
                y1 = (box[1] * self.tile_size) + lon
                x2 = (box[2] * self.tile_size) + lat
                y2 = (box[3] * self.tile_size) + lon
                out_csv.write(f"{x1},{y1},{x2},{y2},{score}\n")
        out_csv.close()

    def torch_image(self, image):
        image = read_image(image)
        resize_transform = transforms.Resize((self.tile_size, self.tile_size))
        image = resize_transform(image)
        image = image.float() / 255.0
        if image.shape[0] == 4:
            image = image[:3, :, :]

        return image

    def test_model(self, threshold=0.25):
        model = torch.load("fishing_model.sav")
        images = os.listdir("grids/")

        # clear file
        prediction_out = open(self.prediction_path, 'w')
        prediction_out.write("x1,y1,x2,y2,score\n")
        prediction_out.close()

        for index, image in enumerate(images):
            name, lat, lon = image.split("_")
            lon = float(lon.split(".")[0])
            lat = float(lat)
            model_image = self.torch_image("grids/" + image)
            self.get_predictions(model_image, model, lat, lon, self.get_location(name), threshold)
            print(f"Image {index}/{len(images)}")

    def draw_predictions(self):

        pred_df = pd.read_csv(self.prediction_path)

        for location in self.locations:
            print("Drawing predictions for " + location.location_name)
            image = Image.open(location.map_path)
            img_width, img_height = image.size
            draw = ImageDraw.Draw(image)
            for coord in pred_df.values:
                x1 = int(coord[0])
                y1 = int(coord[1])
                x2 = min(int(coord[2]), img_width)
                y2 = min(int(coord[3]), img_height)
                score = coord[4]

                draw = ImageDraw.Draw(image)

                rectangle_coords = (int(x1), int(y1), int(x2), int(y2))

                draw.rectangle(rectangle_coords, fill=(int(255 * score * 0.1), 0, 0, int(255 * score)), width=3)

            image.save("pred/" + location.location_name + ".png")
