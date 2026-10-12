
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import shutil
import torch
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--name", default="shape_detector")
    a = ap.parse_args()

    device = a.device if a.device is not None else (
        0 if torch.cuda.is_available() else "cpu"
    )

    print("Training on:", "GPU" if device != "cpu" else "CPU")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    project = os.path.abspath("runs")

    model = YOLO(a.model)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=device,
        workers=2,
        project=project,
        name=a.name,
        exist_ok=True,
        patience=20,
        plots=False,
        amp=False,
        mosaic=0.5,
        scale=0.3,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        degrees=0.0
    )

    os.makedirs("models", exist_ok=True)

    dst = os.path.join("models", "shape_detector.pt")
    best = os.path.join(
        project, a.name, "weights", "best.pt"
    )

    if not os.path.exists(best):
        raise FileNotFoundError(
            f"Training finished, but best.pt was not found: {best}"
        )

    if os.path.exists(dst):
        shutil.copy2(dst, "models/shape_detector.prev.pt")

    shutil.copy2(best, dst)
    print("Saved:", dst)


if __name__ == "__main__":
    main()
