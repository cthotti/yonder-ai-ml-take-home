"""Quick look at the dataset: class counts, box sizes, brightness, and train/valid overlap."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "Sampled-YD-Object-Detection-2"
NAMES = ["bottle", "mallet"]


def source(name):
    # roboflow names copies of the same photo <photo>.rf.<hash>.jpg
    return name.split(".rf.")[0]


for split in ["train", "valid"]:
    images = sorted((DATA / split / "images").glob("*.jpg"))
    counts = Counter()
    areas = {0: [], 1: []}
    brightness = []

    for img_path in images:
        label = DATA / split / "labels" / f"{img_path.stem}.txt"
        for line in label.read_text().splitlines():
            cls, x, y, w, h = line.split()
            counts[int(cls)] += 1
            areas[int(cls)].append(float(w) * float(h))
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        brightness.append(gray.mean())

    print(f"\n{split}: {len(images)} images")
    for c, name in enumerate(NAMES):
        print(f"  {name}: {counts[c]} boxes, median size {np.median(areas[c]) * 100:.1f}% of the image")
    print(f"  median brightness: {np.median(brightness):.0f} / 255")
    print(f"  images darker than 60: {sum(b < 60 for b in brightness)}")

train = {source(p.name) for p in (DATA / "train" / "images").glob("*.jpg")}
valid = [source(p.name) for p in (DATA / "valid" / "images").glob("*.jpg")]
shared = sum(s in train for s in valid)
print(f"\nvalid images whose source photo is also in train: {shared} / {len(valid)}")
