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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "Sampled-YD-Object-Detection-2")
    ap.add_argument("--out", type=Path, default=ROOT / "data_aug")
    ap.add_argument("--copies", type=int, default=1, help="augmented variants per train image")
    ap.add_argument("--seed", type=int, default=0)
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
