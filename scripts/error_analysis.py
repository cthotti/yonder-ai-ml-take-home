"""Find the model's mistakes on the validation set: false positives and missed objects.
Saves every image with a mistake (green = correct, red = false positive, yellow = missed)."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def to_xyxy(x, y, w, h):
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def load_labels(path):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append((int(parts[0]), to_xyxy(*map(float, parts[1:]))))
    return boxes


def match(gts, preds, thr):
    # highest-confidence prediction claims the best unclaimed box of the same class
    used, tps, fps = set(), [], []
    for cls, box, conf in sorted(preds, key=lambda p: -p[2]):
        best, best_iou = None, thr
        for i, (gcls, gbox) in enumerate(gts):
            if i not in used and gcls == cls and iou(box, gbox) >= best_iou:
                best, best_iou = i, iou(box, gbox)
        if best is None:
            fps.append((cls, box, conf))
        else:
            used.add(best)
            tps.append((cls, box, conf))
    fns = [g for i, g in enumerate(gts) if i not in used]
    return tps, fps, fns


def draw(img, tps, fps, fns, names):
    h, w = img.shape[:2]

    def rect(box, color, text):
        x1, y1, x2, y2 = (box * [w, h, w, h]).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    for cls, box, conf in tps:
        rect(box, (0, 200, 0), f"{names[cls]} {conf:.2f}")
    for cls, box, conf in fps:
        rect(box, (0, 0, 255), f"FP {names[cls]} {conf:.2f}")
    for cls, box in fns:
        rect(box, (0, 200, 255), f"FN {names[cls]}")
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "runs" / "aug_v3" / "weights" / "best.pt"))
    parser.add_argument("--data", type=Path, default=ROOT / "data_base")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    model = YOLO(args.weights)
    names = model.names
    img_dir = args.data / "valid" / "images"
    lbl_dir = args.data / "valid" / "labels"
    out = ROOT / "analysis" / "mistakes_valid"
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows, missed, false_pos, confused = [], Counter(), Counter(), Counter()
    for path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        img = cv2.imread(str(path))
        r = model.predict(img, conf=args.conf, imgsz=512, verbose=False)[0]
        preds = [(int(c), to_xyxy(*b.tolist()), float(s))
                 for c, b, s in zip(r.boxes.cls, r.boxes.xywhn, r.boxes.conf)]
        gts = load_labels(lbl_dir / f"{path.stem}.txt")
        tps, fps, fns = match(gts, preds, args.iou)

        for cls, _ in fns:
            missed[names[cls]] += 1
        for cls, box, _ in fps:
            false_pos[names[cls]] += 1
            # a false positive sitting on the other class's object is a mix-up, not background
            if any(g != cls and iou(box, gbox) >= args.iou for g, gbox in gts):
                confused[names[cls]] += 1

        if fps or fns:
            rows.append([path.name, len(gts), len(tps), len(fps), len(fns)])
            cv2.imwrite(str(out / "images" / path.name), draw(img, tps, fps, fns, names))

    rows.sort(key=lambda r: -(r[3] + r[4]))
    with open(out / "mistakes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "objects", "correct", "false_pos", "missed"])
        writer.writerows(rows)

    print(f"\nconf >= {args.conf}, IoU >= {args.iou}")
    print("missed by class:", dict(missed))
    print("false positives by class:", dict(false_pos), " on the other class's object:", dict(confused))
    print(f"images with a mistake: {len(rows)}")
    print("\nimages with the most mistakes:")
    for r in rows[:10]:
        print(f"  fp={r[3]} missed={r[4]}  {r[0]}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
