import os
import json
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import albumentations as A


def resize_with_padding(img, boxes, target_size=256):
    w, h = img.size

    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    img_resized = img.resize((new_w, new_h))

    new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    new_img.paste(img_resized, (pad_x, pad_y)) # Paste img_resized into new_img with top left corner of img_resized at coords (pad_x, pad_y)
    new_boxes = []

    for xmin, ymin, xmax, ymax in boxes: # Coords are between [0:1], they represent the percentage of the widht and height
        # Get the actual coords of the bbox
        xmin = xmin * w
        xmax = xmax * w
        ymin = ymin * h
        ymax = ymax * h

        # Get the new coords for the resized image
        xmin = xmin * scale + pad_x
        xmax = xmax * scale + pad_x
        ymin = ymin * scale + pad_y
        ymax = ymax * scale + pad_y

        # Back to coords between [0:1]
        xmin /= target_size
        xmax /= target_size
        ymin /= target_size
        ymax /= target_size

        new_boxes.append([xmin, ymin, xmax, ymax])

    return new_img, new_boxes


class DigitDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, augment=True):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.augment_enabled = augment

        self.items = []

        for img_name in os.listdir(images_dir):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                base = os.path.splitext(img_name)[0] #os.path.splitext split the file name and the extension
                json_name = base + ".json"

                # Check if the json file exists
                if os.path.exists(os.path.join(annotations_dir, json_name)):
                    self.items.append((img_name, json_name))

        self.augment = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.25,
                    contrast_limit=0.25,
                    p=0.5
                ),

                A.HueSaturationValue(
                    hue_shift_limit=8,
                    sat_shift_limit=15,
                    val_shift_limit=15,
                    p=0.3
                ),

                A.GaussNoise(
                    p=0.2
                ),

                A.MotionBlur(
                    blur_limit=3,
                    p=0.15
                ),

                A.ShiftScaleRotate(
                    shift_limit=0.04,
                    scale_limit=0.12,
                    rotate_limit=7,
                    border_mode=0,
                    value=(255, 255, 255),
                    p=0.5
                ),
            ],
            bbox_params=A.BboxParams(
                format="albumentations",
                label_fields=["class_labels"],
                min_visibility=0.3
            )
        )

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name, ann_name = self.items[idx]

        img_path = os.path.join(self.images_dir, img_name)
        ann_path = os.path.join(self.annotations_dir, ann_name)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        with open(ann_path) as f:
            data = json.load(f)

        boxes = []
        labels = []

        for shape in data["shapes"]:
            (x1, y1), (x2, y2) = shape["points"]

            xmin = min(x1, x2) / w
            ymin = min(y1, y2) / h
            xmax = max(x1, x2) / w
            ymax = max(y1, y2) / h

            boxes.append([xmin, ymin, xmax, ymax])

            # 0 = classe "digit"
            labels.append(0)

        img, boxes = resize_with_padding(img, boxes, target_size=256)

        img_np = np.array(img)

        if self.augment_enabled and len(boxes) > 0:
            augmented = self.augment(
                image=img_np, 
                bboxes=boxes,
                class_labels=labels
            )

            img_np = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["class_labels"]

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        img = self.to_tensor(img_np)

        return img, {"boxes": boxes, "labels": labels}