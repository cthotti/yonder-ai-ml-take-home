"""Adding augmentation methods, blur, noise, constrast, lighting, to the training set. 
Writing a new dataset folder, instead of changing the original download"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def defocus_blur(img):
    k = random.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img):
    # a line kernel rotated to a random angle, like a camera shaking while moving
    k = random.choice([5, 7, 9, 11])
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def noise(img):
    sigma = random.uniform(5, 20)
    out = img.astype(np.float32) + np.random.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast(img):
    # alpha < 1 flattens the image (haze, overcast), alpha > 1 makes it harsher
    alpha = random.uniform(0.6, 1.6)
    mean = img.mean()
    out = (img.astype(np.float32) - mean) * alpha + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma(img):
    # g > 1 darkens shadows, g < 1 lifts them
    g = random.uniform(0.5, 2.0)
    table = ((np.arange(256) / 255.0) ** g * 255).astype(np.uint8)
    return cv2.LUT(img, table)

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

AUGS = [defocus_blur, motion_blur, noise, contrast, gamma, grayscale]


def augment(img):
    picked = random.sample(AUGS, random.randint(1, 2))
    for fn in picked:
        img = fn(img)
    return img, [fn.__name__ for fn in picked]

def read_boxes(label_path, margin=0.02):
    # labeled boxes as x1, y1, x2, y2 (normalized), padded so crops stay clear of objects
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                x, y, w, h = map(float, p[1:])
                boxes.append((x - w / 2 - margin, y - h / 2 - margin, x + w / 2 + margin, y + h / 2 + margin))
    return boxes


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def background_crop(img, boxes, tries=50):
    # cut a square that doesn't touch any object, scaled back to full size
    h, w = img.shape[:2]
    for _ in range(tries):
        size = int(random.uniform(0.4, 0.7) * min(h, w))
        x, y = random.randint(0, w - size), random.randint(0, h - size)
        crop = (x / w, y / h, (x + size) / w, (y + size) / h)
        if not any(overlaps(crop, b) for b in boxes):
            return cv2.resize(img[y:y + size, x:x + size], (w, h))
    return None




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "Sampled-YD-Object-Detection-2")
    ap.add_argument("--out", type=Path, default=ROOT / "data_aug")
    ap.add_argument("--copies", type=int, default=1, help="augmented variants per train image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backgrounds", type=int, default=0, help="object-free crops to add")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.out.exists():
        shutil.rmtree(args.out)

    # validation stays untouched so metrics reflect the real images
    shutil.copytree(args.src / "valid", args.out / "valid")

    img_out = args.out / "train" / "images"
    lbl_out = args.out / "train" / "labels"
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    made = 0
    for img_path in sorted((args.src / "train" / "images").iterdir()):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl_path = args.src / "train" / "labels" / f"{img_path.stem}.txt"

        shutil.copy(img_path, img_out / img_path.name)
        if lbl_path.exists():
            shutil.copy(lbl_path, lbl_out / lbl_path.name)

        # pixels don't move, so the original boxes are still correct
        for i in range(args.copies):
            aug, names = augment(img)
            stem = f"{img_path.stem}_aug{i}_{'+'.join(names)}"
            cv2.imwrite(str(img_out / f"{stem}.jpg"), aug)
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_out / f"{stem}.txt")
            made += 1
    
    # backgrounds: object-free crops of the original photos, with empty label files
    train_imgs = sorted((args.src / "train" / "images").iterdir())
    added = 0
    for img_path in random.sample(train_imgs, len(train_imgs)):
        if added == args.backgrounds:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        crop = background_crop(img, read_boxes(args.src / "train" / "labels" / f"{img_path.stem}.txt"))
        if crop is None:
            continue
        cv2.imwrite(str(img_out / f"{img_path.stem}_bg.jpg"), crop)
        (lbl_out / f"{img_path.stem}_bg.txt").write_text("")
        added += 1
    print(f"Added {added} background crops")

    names = yaml.safe_load((args.src / "data.yaml").read_text())["names"]
    data = {
        "path": str(args.out.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(names),
        "names": names,
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    print(f"Added {made} augmented images to {img_out}")
    print(f"Dataset config: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
