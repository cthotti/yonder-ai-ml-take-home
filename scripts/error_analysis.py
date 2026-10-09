"""Find the model's mistakes on a split: false positives, false negatives and low-confidence hits.
Run on valid for the error analysis, and on train for hard example mining."""
import argparse
import csv
from collections import Counter, defaultdict
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


def on_other_class(fp, gts, thr):
    # an FP sitting on the other class's box is a confusion, not a background hit
    return any(gcls != fp[0] and iou(fp[1], gbox) >= thr for gcls, gbox in gts)


def aug_tag(stem):
    # data_aug names look like <photo>_aug0_blur+gamma; originals have no tag
    return stem.split("_aug", 1)[1].split("_", 1)[1] if "_aug" in stem else "original"


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
    parser.add_argument("--weights", default=str(ROOT / "runs" / "aug_v2" / "weights" / "best.pt"))
    parser.add_argument("--data", type=Path, default=ROOT / "data_aug")
    parser.add_argument("--split", default="valid", choices=["valid", "train"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=512)
    args = parser.parse_args()

    model = YOLO(args.weights)
    names = model.names
    img_dir = args.data / args.split / "images"
    lbl_dir = args.data / args.split / "labels"
    out = ROOT / "analysis" / f"mistakes_{args.split}"
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows, fn_count, fp_count, confused = [], Counter(), Counter(), Counter()
    for path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        img = cv2.imread(str(path))
        r = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        preds = [(int(c), to_xyxy(*b.tolist()), float(s))
                 for c, b, s in zip(r.boxes.cls, r.boxes.xywhn, r.boxes.conf)]
        gts = load_labels(lbl_dir / f"{path.stem}.txt")
        tps, fps, fns = match(gts, preds, args.iou)

        for cls, _ in fns:
            fn_count[names[cls]] += 1
        for fp in fps:
            fp_count[names[fp[0]]] += 1
            if on_other_class(fp, gts, args.iou):
                confused[names[fp[0]]] += 1

        # hardness: every mistake counts 1, plus how unsure it was on the boxes it got right
        min_conf = min((t[2] for t in tps), default=1.0)
        score = len(fps) + len(fns) + (1 - min_conf)
        rows.append([path.name, aug_tag(path.stem), len(gts), len(tps), len(fps), len(fns),
                     round(min_conf, 3), round(score, 3)])

        if fps or fns:
            cv2.imwrite(str(out / "images" / path.name), draw(img, tps, fps, fns, names))

    rows.sort(key=lambda r: -r[-1])
    with open(out / "mistakes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "aug", "n_gt", "tp", "fp", "fn", "min_tp_conf", "hardness"])
        writer.writerows(rows)

    print(f"\n{args.split}: {len(rows)} images, conf >= {args.conf}, IoU >= {args.iou}")
    print("false negatives by class:", dict(fn_count))
    print("false positives by class:", dict(fp_count), " on the other class's box:", dict(confused))
    print(f"images with a mistake: {sum(r[4] + r[5] > 0 for r in rows)}")

    print("\nhardest images:")
    for r in rows[:10]:
        print(f"  {r[-1]:5.2f}  fp={r[4]} fn={r[5]} min_conf={r[6]:.2f}  {r[0]}")

    # which conditions the model struggles with most (useful on the train split)
    by_tag = defaultdict(list)
    for r in rows:
        for tag in r[1].split("+"):
            by_tag[tag].append(r)
    if len(by_tag) > 1:
        print(f"\n{'condition':<14} {'images':>6} {'error rate':>10} {'mean hardness':>14}")
        for tag, rs in sorted(by_tag.items(), key=lambda kv: -np.mean([r[-1] for r in kv[1]])):
            err = np.mean([r[4] + r[5] > 0 for r in rs])
            print(f"{tag:<14} {len(rs):>6} {err:>10.3f} {np.mean([r[-1] for r in rs]):>14.3f}")

    print(f"\nAnnotated mistakes and mistakes.csv in {out}")


if __name__ == "__main__":
    main()
