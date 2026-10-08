import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def pick_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data_aug" / "data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="aug")
    args = parser.parse_args()

    device = pick_device()
    print("training on", device)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        seed=0,
        hsv_h=0.1,
        patience=10,
    )

    # evaluate the best checkpoint on the validation set, per class
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=device)
    box = metrics.box

    print(f"\n{'class':<8} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}")
    for i, cls in enumerate(box.ap_class_index):
        name = model.names[int(cls)]
        print(f"{name:<8} {box.p[i]:6.3f} {box.r[i]:6.3f} {box.ap50[i]:7.3f} {box.ap[i]:9.3f}")
    print(f"{'all':<8} {box.mp:6.3f} {box.mr:6.3f} {box.map50:7.3f} {box.map:9.3f}")


if __name__ == "__main__":
    main()
