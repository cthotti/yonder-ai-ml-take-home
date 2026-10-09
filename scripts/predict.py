"""Run the detector on a folder of images.
Writes one .txt per image: class_id x_center y_center width height confidence (normalized)."""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--weights", default=str(ROOT / "runs" / "aug" / "weights" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=512)
    args = parser.parse_args()

    model = YOLO(args.weights)
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXTS)
    total = 0
    for path in paths:
        result = model.predict(str(path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = result.boxes

        lines = []
        for cls, xywh, conf in zip(boxes.cls, boxes.xywhn, boxes.conf):
            x, y, w, h = xywh.tolist()
            lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {float(conf):.4f}")

        # an empty file means nothing was found above the threshold
        (args.out / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        total += len(lines)

    print(f"{len(paths)} images, {total} detections at conf >= {args.conf}, written to {args.out}")


if __name__ == "__main__":
    main()
