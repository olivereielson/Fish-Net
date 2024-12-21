import os
from torchvision.transforms import v2

import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from final.dataset import Map_Dataset
import torchvision.transforms.v2 as T
import pandas as pd
from torch.utils.data import Dataset

from final.train_helper import collate_fn


def get_color(score, alpha):
    cmap = plt.get_cmap('RdYlGn')  # Choose a colormap (you can change it)
    rgba_color = cmap(score)
    return tuple(int(c * 255) for c in rgba_color[:3]) + (alpha,)


def highlight_box(image, box, alpha, corner_radius=10):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Use light purple color (RGB: 220, 200, 255) with the specified alpha value
    color = (244, 117, 73, alpha)

    # Convert box coordinates to integers
    box = [int(coord) for coord in box]

    # Use rounded_rectangle from ImageDraw
    ImageDraw.Draw(overlay).rounded_rectangle(box, fill=color, radius=corner_radius)

    # Combine the overlay with the original image
    image = Image.alpha_composite(image.convert("RGBA"), overlay)

    # Convert back to RGB
    return image.convert("RGB")


print("******Preparing Envoirment******")

transform = T.Compose([
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

])

device = torch.device('cpu')

print("******Preparing Data******")
files = os.listdir("/Users/olivereielson/Desktop/stolen_goods")

print("******Preparing Model******")
model = torch.load("/Users/olivereielson/Desktop/fishing_model-3.sav", map_location=torch.device('cpu'))
model.to(device)
model.eval()
number_shown = 0

for images in files:

    print(f"image {number_shown}/{len(files)}")
    if images == ".DS_Store":
        continue

    image = Image.open("/Users/olivereielson/Desktop/stolen_goods/" + images).convert("RGB")
    transformed_image = transform(image).to(device)  # Use a different variable name
    predictions = model([transformed_image])

    boxes = predictions[0]["boxes"].tolist()
    scores = predictions[0]["scores"].tolist()

    t = T.ToPILImage()
    image = t(image)

    draw = ImageDraw.Draw(image)
    for score, box in zip(scores, boxes):
        # print(box)
        # box = [b * 300 for b in box]
        if score < 0.2:
            continue
        color = get_color(score, 128)
        # draw.rectangle(box, outline=color, width=5)
        image = highlight_box(image, box, 150, corner_radius=15)  # Set alpha value for the highlighted box

    #image.show()
    image.save("/Users/olivereielson/Desktop/out/" + images)
    number_shown += 1