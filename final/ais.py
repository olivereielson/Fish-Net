from datetime import datetime, timedelta

import requests
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import os

bad_MMSI = [338356427.0, 338226496.0, 367459470, 368104650, 367096960, 368128140, 338168958,
            367393710, 367638310, 338203701, 338229321, 368124390, 338229941, 8704092, 368128140, 367526270, 367072720,
            368006440, 367105250,338356627,367101090,367518040]
def get_new_boats():
    df = pd.read_csv('new_file.csv')

    df = df[df['VesselType'] == 30]
    df = df[df['Length'] < 14]
    df = df[df['SOG'] < 5]

    for mmsi in bad_MMSI:
        df = df[df['MMSI'] != mmsi]

    df = df[df['MMSI'] != 338356427.0]
    df = df[df['MMSI'] != 338226496.0]

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LON, df.LAT))

    newport_coords = (41.12807, -71.88187)  # Replace with the actual coordinates

    filtered_gdf = gdf[gdf.geometry.apply(lambda point: geodesic(newport_coords, (point.y, point.x)).miles <= 60)]

    return filtered_gdf


def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)


def unzip_file(zip_file_path, extract_to, new_file_name):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Assuming there is only one file in the zip archive
    # You might need to modify this logic based on your specific case
    original_file_path = zip_ref.namelist()[0]
    extracted_file_path = os.path.join(extract_to, original_file_path)

    # Rename the extracted file
    new_file_path = os.path.join(extract_to, new_file_name)
    os.rename(extracted_file_path, new_file_path)


def get_new_data(url):
    file_name = 'new_file.zip'

    response = requests.get(url)

    with open(file_name, 'wb') as file:
        file.write(response.content)

    print(f'The file {file_name} has been downloaded.')


# Example usage:

start_date = datetime(2022, 8, 1)

# Loop through each day for the next three months
for month_offset in range(3):
    current_date = start_date + timedelta(days=30 * month_offset)

    for day in range(1, 32):
        try:
            current_url = f'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2022/AIS_{current_date.strftime("%Y_%m")}_{day:02d}.zip'
            file_name = f'AIS_{current_date.strftime("%Y_%m_%d")}.zip'

            print("Downloading file: " + current_url)
            get_new_data(current_url)

            print("Unzipping file")
            unzip_file('new_file.zip', './', "new_file.csv")

            print("Getting new boats")
            new_data = get_new_boats()

            print("Merging data")
            if os.path.exists("all.csv"):
                old_data = pd.read_csv('all.csv')
                merged_df = pd.concat([new_data, old_data], ignore_index=True)
                merged_df.to_csv('all.csv', index=False)
            else:
                new_data.to_csv('all.csv', index=False)
        except:
            print("File not found")

        print("Deleting files")
        if os.path.exists("new_file.zip"):
            os.remove("new_file.zip")

        if os.path.exists("new_file.csv"):
            os.remove("new_file.csv")
