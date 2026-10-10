"""Export the final model, quantize it to INT8, and compare size, accuracy and CPU latency.
CPU timing here is a stand-in for the rover; real NPU (RKNN) numbers will differ."""
import argparse
import csv
from pathlib import Path
import shutil
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_base" / "data.yaml"


def size_mb(path):
    path = Path(path)
    files = [path] if path.is_file() else [f for f in path.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


def latency_ms(model, images, imgsz):
    for img in images[:5]:
        model.predict(str(img), imgsz=imgsz, device="cpu", verbose=False)  # warm up
    times = [model.predict(str(img), imgsz=imgsz, device="cpu", verbose=False)[0].speed["inference"]
             for img in images]
    return float(np.median(times))


def evaluate(name, path, imgsz, images):
    model = YOLO(str(path), task="detect")
    box = model.val(data=str(DATA), imgsz=imgsz, device="cpu", plots=False, verbose=False,
                    project=str(ROOT / "runs" / "efficiency"), name=name.replace(" ", "_"), exist_ok=True).box
    recall = dict(zip(box.ap_class_index.tolist(), box.r.tolist()))
    return [name, round(size_mb(path), 1), round(box.map50, 3), round(box.map, 3),
            round(recall.get(1, 0), 3), round(latency_ms(model, images, imgsz), 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "runs" / "aug_v2" / "weights" / "best.pt")
    args = parser.parse_args()

    # calibrate INT8 on training images so the validation set stays unseen
    cfg = yaml.safe_load(DATA.read_text())
    cfg["val"] = "train/images"
    calib = DATA.parent / "calib.yaml"
    calib.write_text(yaml.safe_dump(cfg))

    model = YOLO(str(args.weights))
    print(f"parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    onnx_path = YOLO(str(args.weights)).export(format="onnx", imgsz=512)
    ov_fp32 = YOLO(str(args.weights)).export(format="openvino", imgsz=512)
    ov_int8 = YOLO(str(args.weights)).export(format="openvino", imgsz=512, int8=True, data=str(calib))
    
    # separate copy so the 416 export doesn't overwrite the 512 one
    w416 = args.weights.with_name("best_416.pt")
    shutil.copy(args.weights, w416)
    ov_int8_416 = YOLO(str(w416)).export(format="openvino", imgsz=416, int8=True, data=str(calib))
    
    et_512 = YOLO(str(args.weights)).export(format="executorch", imgsz=512)
    et_416 = YOLO(str(w416)).export(format="executorch", imgsz=416)

    images = sorted((ROOT / "data_base" / "valid" / "images").iterdir())[:50]
    variants = [
        ("pytorch fp32 512", args.weights, 512),
        ("pytorch fp32 416", args.weights, 416),
        ("pytorch fp32 320", args.weights, 320),
        ("onnx fp32 512", onnx_path, 512),
        ("openvino fp32 512", ov_fp32, 512),
        ("openvino int8 512", ov_int8, 512),
        ("openvino int8 416", ov_int8_416, 416),
        ("executorch fp32 512", et_512, 512),
        ("executorch fp32 416", et_416, 416),
    ]
    rows = [evaluate(name, path, imgsz, images) for name, path, imgsz in variants]

    header = ["variant", "size_MB", "mAP50", "mAP50-95", "R_mallet", "cpu_ms"]
    print(f"\n{header[0]:<19} " + " ".join(f"{h:>9}" for h in header[1:]))
    for r in rows:
        print(f"{r[0]:<19} " + " ".join(f"{v:>9}" for v in r[1:]))

    out = ROOT / "analysis" / "efficiency.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
