"""Download the Yonder mallet/bottle dataset from Roboflow.

Reads the API key from the ROBOFLOW_API_KEY environment variable (or a git-ignored .env).
Writes to Sampled-YD-Object-Detection-2/ at the repo root. Treat that folder as read-only.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Sampled-YD-Object-Detection-2"


def main():
    load_dotenv(ROOT / ".env")
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("ROBOFLOW_API_KEY is not set. Put it in .env or export it.")

    if (DATASET_DIR / "data.yaml").exists():
        print(f"Dataset already at {DATASET_DIR}, skipping download.")
        return

    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    project = rf.workspace("malletbottle2").project("sampled-yd-object-detection")
    project.version(2).download("yolov8", location=str(DATASET_DIR))
    print(f"Downloaded to {DATASET_DIR}")


if __name__ == "__main__":
    main()
