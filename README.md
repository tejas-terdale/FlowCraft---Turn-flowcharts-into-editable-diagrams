# FlowCraft - flowchart image to editable draw.io diagram

Upload a clean black-on-white flowchart (rectangles and diamonds joined by straight arrows) and get a
`.drawio` file. Open it at https://app.diagrams.net (File > Open) and edit it like any diagram: every shape is
separate, every label is editable text, and every arrow is a real connector that follows the shapes when you
move them.

## How it works

1. **Shape detection - deep learning.** A YOLOv8n network (`models/shape_detector.pt`), trained on synthetic
   flowcharts, finds every shape and classifies it as `rect` or `diamond`. Each box is then snapped to the
   exact outline in the pixels, so the later steps stay pixel-accurate.
   *Fallback:* a classical OpenCV method (contours, fill ratio, corner count) finds the same shapes without any
   neural network.
2. **Text.** Tesseract OCR reads each label, using only the pixels inside that shape.
3. **Edges.** All shapes are erased from the image; every remaining line is one edge, and the two shapes it
   touches are its ends. A line that touches anything other than exactly two shapes is skipped.
4. **Arrow direction.** The end of a line that is much thicker than the rest of the line (more than 1.3x) is
   the arrowhead.
5. **Output.** A `.drawio` file with rounded rectangles, diamonds and directed connectors, in the original layout.

Shape detection is the only step where the two methods differ; steps 2-5 are shared.

## Detection modes

| Mode (web page) | `--method` | What runs |
|---|---|---|
| Auto Detect (default) | `auto` | YOLO; falls back to classical if the weights are missing, YOLO crashes, or it finds nothing (prints a message) |
| YOLO | `dl` | YOLO only. **Never falls back**: missing weights or a crash is an error, and no detections means 0 shapes |
| Classical | `classical` | Classical only. YOLO is never loaded |

`python test_modes.py` proves this by spying on both detectors and sabotaging YOLO on purpose (12 checks).

## Quick start

```
pip install -r requirements.txt
python app.py                      # open http://127.0.0.1:5000, upload, download diagram.drawio
```

Install Tesseract OCR itself too: Windows `winget install UB-Mannheim.TesseractOCR` (if Python cannot find it,
set `pytesseract.pytesseract.tesseract_cmd` at the top of `flow2drawio.py`), macOS `brew install tesseract`,
Linux `sudo apt install tesseract-ocr`.

The page processes everything locally. On start it converts `flowchart2.png` once as a warm-up and prints
`warm-up OK (detector: dl)`.

From the terminal, without the web page:

```
python flow2drawio.py flowchart2.png out.drawio --method auto      # or: dl / classical
```

## Using your own trained model (CPU or GPU, no code change)

The converter always loads `models/shape_detector.pt`. To switch models, put the one you want under that name
and restart the app. Or point to any file with the environment variable `FLOWCHART_MODEL`
(PowerShell: `$env:FLOWCHART_MODEL="C:\path\model.pt"`). Where a model was trained does not matter at inference;
PyTorch uses the GPU automatically when it can see one. The inference image size is read from the model file.

## Training

```
# 1. Windows: plain "pip install torch" gives a CPU-only build. Install the CUDA build first, e.g.
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#    and check:  python -c "import torch; print(torch.cuda.is_available())"   -> must print True

# 2. Generate synthetic training data with exact labels (needs an empty/new --out folder)
python gen_data.py --out data --train 3000 --val 300 --test 100

# 3. Train (this is the setup used on an RTX 4050 6 GB; lower --batch to 8 or 4 if you run out of memory)
python train.py --epochs 20 --imgsz 640 --batch 16

# 4. Compare classical and deep learning on held-out charts with known answers
python eval_pipeline.py --method both
```

`train.py` saves the best checkpoint to `models/shape_detector.pt` and first copies any existing model to
`models/shape_detector.prev.pt`. `gen_data.py` refuses to write into a folder that already has files, so it
cannot overwrite an existing dataset. Its data format is YOLO labels plus a ground-truth JSON per chart
(`images/`, `labels/`, `gt/`, `data.yaml`).

Useful options: `eval_pipeline.py --data data2 --limit 30 --perturb 1` (`--perturb 1|2` adds blur, noise and JPEG
compression before converting).

## Results

All numbers below were measured, not estimated. Test charts are synthetic with exact ground truth.

**Training runs**

| | CPU run | GPU run |
|---|---|---|
| Data | 299 charts | 3,000 train / 300 val charts |
| Settings | 8 epochs, 512 px, batch 8 | 20 epochs, 640 px, batch 16, amp off |
| Hardware | Xeon CPU | RTX 4050 Laptop GPU (6 GB) |
| Time | about 13 min | about 14 min (37-46 s per epoch) |
| Validation | P 1.00, R 1.00, mAP50 0.995, mAP50-95 0.988 | P 1.00, R 1.00, mAP50 0.995, mAP50-95 0.995 |

Validation scores hit the ceiling on this easy synthetic data, so they show the network learns the task, not
that it works on every flowchart.

**Full pipeline on 60 held-out synthetic charts (CPU model)**

| pipeline | node F1 | label acc | edge F1 (directed) | edge F1 (connection) | charts fully correct |
|---|---|---|---|---|---|
| classical | 1.000 | 0.997 | 0.951 | 0.997 | 41/60 |
| deep learning | 1.000 | 0.997 | 0.951 | 0.997 | 41/60 |

On a second, harder synthetic test set (many lines touching or crossing other shapes) shape detection is still
about 0.99+ F1 but only about 73% of connections are recovered, for both methods.

On blurred, noisy, JPEG-compressed charts (30 charts) shapes and connections stay perfect for both methods;
only label accuracy drops (0.967 moderate, 0.428 severe), because the limit is OCR.

Both demo images (`flowchart.png`, `flowchart2.png`) are reproduced exactly in every mode:
`python test_ground_truth.py`.

**Honest note:** on rectangles and diamonds the deep-learning detector ties the classical rules. Its advantage is
extensibility: a new shape class (oval, parallelogram...) needs new generated data and retraining, not new
hand-written geometry rules. Test-set numbers for a model you train yourself are not included here; run
`python eval_pipeline.py --method both` to measure yours.

## Troubleshooting

- **Browser shows ERR_CONNECTION_RESET:** the Python process crashed. Read the terminal where you ran
  `python app.py`; the real error is printed there. `app.py` imports PyTorch first and warms the model up at start
  so a crash happens at launch with a message. If warm-up fails, choose "Classical" in the page, which needs no
  PyTorch.
- **Isolate a problem without the web page:**
  `python flow2drawio.py flowchart2.png out.drawio --method classical` and then `--method dl`. If classical works
  and dl crashes, the problem is the PyTorch install, not this project.
- **Training says CPU:** you have a CPU-only PyTorch; install the CUDA build (see Training).
- **CUDA out of memory:** lower `--batch` (try 8, then 4) or `--imgsz`.
- **Windows training errors about workers:** `train.py` already wraps training in `main()` under
  `if __name__ == "__main__"`, which Windows needs for dataloader workers.
- **Tesseract not found:** install it and set `pytesseract.pytesseract.tesseract_cmd` (see Quick start).

## Known limits

- Clean, printed-style, high-contrast charts only; no handwriting, photos or heavy degradation.
- Only rectangles and diamonds; no ovals, parallelograms or other shapes.
- No text on edges (for example "Yes"/"No" labels); edges are straight lines (draw.io re-routes them if you move
  shapes); elbow lines are untested.
- A line touching more than two shapes (a branch or merge drawn as one line, or a line running through a third
  shape) is skipped with the message `skipped a line touching N shapes`. This is the biggest measured cause of
  missed edges.
- Arrow direction is wrong on roughly 5% of edges in the main synthetic test set.
- About 2 s per image on a 2-core CPU, mostly OCR; YOLO inference itself takes a few milliseconds on a GPU.
- Training and test data are synthetic and from one visual style; other drawing tools are untested.

## Project files

| File | Purpose |
|---|---|
| `app.py` | FlowCraft web app (Flask) |
| `flow2drawio.py` | The converter: both detectors, OCR, edges, arrows, draw.io writer, command line |
| `gen_data.py` | Synthetic flowchart generator (YOLO labels + ground-truth JSON) |
| `train.py` | YOLO training script |
| `eval_pipeline.py` | Classical vs deep learning on ground-truth charts, optional image degradation |
| `test_ground_truth.py` | Regression test on the two demo images, both modes |
| `test_modes.py` | Proves each mode runs what it claims |
| `models/shape_detector.pt` | The deployed model (`.prev.pt` = the one it replaced) |
| `data/`, `runs/` | Dataset and training outputs |
| `flowchart.png`, `flowchart2.png`, `demo_*.drawio` | Demo images and their converted output |

Note: `gen_data.py` was rebuilt to match the original command-line interface and data layout after the first
version was overwritten, so charts it makes are simpler (axis-aligned edges between neighbouring boxes) than
the earlier datasets.
