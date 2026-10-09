"""Compare baseline vs augmented weights on corrupted copies of the validation set.
Each condition applies one fixed corruption, so both models see identical images."""
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from augment import grayscale
from train import pick_device

ROOT = Path(__file__).resolve().parent.parent
VALID = ROOT / "data_base" / "valid"
OUT = ROOT / "data_stress"

MODELS = {
    "baseline": ROOT / "runs" / "baseline" / "weights" / "best.pt",
    "aug": ROOT / "runs" / "aug" / "weights" / "best.pt",
    "aug_v2": ROOT / "runs" / "aug_v2" / "weights" / "best.pt",
    "aug_v3": ROOT / "runs" / "aug_v3" / "weights" / "best.pt",
}


def gamma(img, g):
    table = ((np.arange(256) / 255.0) ** g * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def noise(img, sigma):
    rng = np.random.default_rng(0)
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


CONDITIONS = {
    "clean": lambda img: img,
    "blur": lambda img: cv2.GaussianBlur(img, (9, 9), 0),
    "dark": lambda img: gamma(img, 2.0),
    "low_contrast": lambda img: cv2.convertScaleAbs(img, alpha=0.5, beta=64),
    "noise": lambda img: noise(img, 20),
    "grayscale": grayscale,
}


def build(name, fn, names):
    folder = OUT / name
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "images").mkdir(parents=True)
    shutil.copytree(VALID / "labels", folder / "labels", ignore=shutil.ignore_patterns("*.cache"))

    # saved as png so the clean set doesn't pick up extra jpeg loss
    for path in sorted((VALID / "images").iterdir()):
        img = cv2.imread(str(path))
        if img is not None:
            cv2.imwrite(str(folder / "images" / f"{path.stem}.png"), fn(img))

    cfg = folder / "data.yaml"
    cfg.write_text(yaml.safe_dump({"path": str(folder), "train": "images", "val": "images", "names": names}))
    return cfg


def main():
    names = yaml.safe_load((ROOT / "data_base" / "data.yaml").read_text())["names"]
    models = {name: YOLO(str(path)) for name, path in MODELS.items()}
    device = pick_device()

    rows = []
    for cond, fn in CONDITIONS.items():
        cfg = build(cond, fn, names)
        for mname, model in models.items():
            box = model.val(data=str(cfg), imgsz=512, device=device, plots=False, verbose=False,
                            project=str(ROOT / "runs" / "stress"), name=f"{mname}_{cond}", exist_ok=True).box
            recall = dict(zip(box.ap_class_index.tolist(), box.r.tolist()))
            rows.append([cond, mname, round(box.map50, 3), round(box.map, 3),
                         round(recall.get(0, 0), 3), round(recall.get(1, 0), 3)])

    header = ["condition", "model", "mAP50", "mAP50-95", "R_bottle", "R_mallet"]
    print(f"\n{header[0]:<13} {header[1]:<9} " + " ".join(f"{h:>9}" for h in header[2:]))
    for r in rows:
        print(f"{r[0]:<13} {r[1]:<9} " + " ".join(f"{v:>9.3f}" for v in r[2:]))

    out_csv = ROOT / "analysis" / "stress_results.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
