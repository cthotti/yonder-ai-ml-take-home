"""Live detection from a webcam. Press s to save the current frame, q to quit.
Raw frames go to <out>/raw so predict.py can be run on them afterwards."""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "runs" / "aug" / "weights" / "best.pt"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--out", type=Path, default=ROOT / "video_test")
    args = parser.parse_args()

    raw_dir = args.out / "raw"
    ann_dir = args.out / "annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"couldn't open camera {args.camera}, try --camera 1")

    saved = len(list(raw_dir.glob("*.jpg")))
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        start = time.time()
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        ms = (time.time() - start) * 1000

        shown = result.plot()
        cv2.putText(shown, f"{ms:.0f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("live", shown)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = f"frame_{saved:03d}.jpg"
            cv2.imwrite(str(raw_dir / name), frame)
            cv2.imwrite(str(ann_dir / name), shown)
            print("saved", name)
            saved += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
