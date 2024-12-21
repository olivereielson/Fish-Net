import math
import os
import random
import sys
import time
from torchmetrics.detection import MeanAveragePrecision
from sklearn.model_selection import train_test_split

from torchvision import transforms
import torchvision
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
import torch.nn.functional as F


def split_data(data, test_size=0.10, valid_size=0.2, random_state=40):
    unique_image_paths = data['image_path'].unique()
    train_paths, temp_paths = train_test_split(unique_image_paths, test_size=(test_size + valid_size),
                                               random_state=random_state)
    test_paths, valid_paths = train_test_split(temp_paths, test_size=valid_size / (test_size + valid_size),
                                               random_state=random_state)

    train_data = data[data['image_path'].isin(train_paths)].copy()
    test_data = data[data['image_path'].isin(test_paths)].copy()
    valid_data = data[data['image_path'].isin(valid_paths)].copy()

    return train_data, test_data, valid_data


def get_gpu_status():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well. (From PyTorch documentation)
    """
    return tuple(zip(*batch))


def eval_mAP(model, dataloader, device, metric):
    model.eval()
    all_avg_precisions = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(images)
            preds = [{k: v.to(device) for k, v in t.items()} for t in preds]

            formatted_preds = [{'labels': p['labels'], 'boxes': p['boxes'], 'scores': p['scores']} for p in preds]
            formatted_targets = [{'labels': t['labels'], 'boxes': t['boxes']} for t in targets]

            metric.update(formatted_preds, formatted_targets)
            avg_precision = metric.compute()['map']

            all_avg_precisions.append(avg_precision)

    average_mAP = sum(all_avg_precisions) / len(all_avg_precisions)

    return average_mAP


def Eval_loss(model, dataloader, device):
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / len(dataloader)


def train_one_epoch(model, optimizer, data_loader, device, lr_scheduler):
    # set model to training mode
    model.train()

    itr = 0
    total_loss = 0.0
    for images, targets in data_loader:
        # load data onto the GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # train on data
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()

        # check if I was messed something up
        if not math.isfinite(losses.item()):
            print(f"Loss is {losses.item()}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return total_loss / len(data_loader)


def show_examples(model, dataloader, device, num_examples=5):
    """
    Show `num_examples` predictions from the dataset.
    """
    model.eval()
    number_shown = 0
    for images, targets in dataloader:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        predictions = model(images)

        for image, prediction, target in zip(images, predictions, targets):
            boxes = prediction["boxes"].tolist()
            boxes_real = target["boxes"].tolist()
            scores = prediction["scores"].tolist()

            t = transforms.ToPILImage()
            image = t(image)
            draw = ImageDraw.Draw(image)
            for box in boxes:
                draw.rectangle(box, outline="red", width=3)

            for box in boxes_real:
                draw.rectangle(box, outline="green", width=3)

            image.show()
            number_shown += 1
            if number_shown >= num_examples:
                return
