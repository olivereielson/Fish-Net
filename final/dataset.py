import torch

from PIL import Image, ImageDraw
from torch.utils.data import Dataset

import torchvision.transforms.v2 as transforms


class Map_Dataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None, max_boxes=100, tile_size=None, tile_size_w=1000,
                 tile_size_h=685):
        super().__init__()
        # Save the passed in DayaSet parameters
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.tile_size = tile_size
        self.tile_size_w = tile_size_w
        self.tile_size_h = tile_size_h

        self.images = []
        self.labels = []
        self.boxes = []
        self.image_paths = []

        grouped = self.data.groupby('image_path')

        for image, data in grouped:

            boxes = []
            labels = []
            self.image_paths.append(image)

            for row in data.iterrows():
                labels.append(row[1].iloc[1])

                row[1].iloc[2] = max(row[1].iloc[2], 0)
                row[1].iloc[3] = max(row[1].iloc[3], 0)

                row[1].iloc[4] = min(row[1].iloc[4], 1000)
                row[1].iloc[5] = min(row[1].iloc[5], 685)

                boxes.append([(row[1].iloc[2]), (row[1].iloc[3]), (row[1].iloc[4]),
                              (row[1].iloc[5])])
                # boxes.append([(row[1].iloc[2] / tile_size), (row[1].iloc[3] / tile_size), (row[1].iloc[4] / tile_size),
                #               (row[1].iloc[5] / tile_size)])

            # Pad or truncate boxes to a fixed size
            if len(boxes) < max_boxes:
                pad_size = max_boxes - len(boxes)
                boxes += [[0, 0, 999, 685]] * pad_size
                labels += [0] * pad_size  # Use a placeholder label for padded boxes
            else:
                boxes = boxes[:max_boxes]
                labels = labels[:max_boxes]

            self.boxes.append(boxes)
            self.labels.append(labels)

    def __len__(self):
        return len(self.image_paths)

    def show_image(self, idx, image, boxes):

        to_pil = transforms.ToPILImage()
        pil_image = to_pil(image)

        draw = ImageDraw.Draw(pil_image)
        for coord in boxes:
            x1 = int(coord[0])
            y1 = int(coord[1])
            x2 = min(int(coord[2]), self.tile_size_w)
            y2 = min(int(coord[3]), self.tile_size_h)
            rectangle_coords = (int(x1), int(y1), int(x2), int(y2))
            draw.rectangle(rectangle_coords, outline=(0, 255, 0), width=3, )

        pil_image.show()

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        boxes = torch.as_tensor(boxes, dtype=torch.int)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.zeros((len(labels)), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([idx])

        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        # if image != image2:
        #     assert boxes2 != boxes

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target
