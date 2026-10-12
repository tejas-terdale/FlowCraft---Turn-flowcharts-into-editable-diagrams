# Flowchart image -> editable draw.io diagram

Upload a black-on-white flowchart (rectangles, diamonds, straight/diagonal arrows) and get a `.drawio` file
you can edit at https://app.diagrams.net .

## How it works
1. **Shape detection (deep learning):** a YOLOv8 network (`models/shape_detector.pt`), trained on synthetic
   flowcharts, finds every shape and classifies it as `rect` or `diamond`. Each box is then snapped to the exact
   outline in the image. *Fallback:* if the weights or ultralytics are missing, a classical OpenCV contour method is used.
2. **Text:** Tesseract OCR reads each label (masked to the inside of its own shape).
3. **Edges:** shapes are erased, so every remaining line component is one edge; the two shapes it touches are its ends.
4. **Arrow direction:** the end of a line that is "fatter" than the rest of the line is the arrowhead.
5. A `.drawio` file is written with rounded rectangles, diamonds and directed edges.

## Run it
    pip install -r requirements.txt          # + install Tesseract OCR itself (see below)
    python app.py                            # open http://127.0.0.1:5000, upload, download diagram.drawio
    python flow2drawio.py flowchart2.png out.drawio --method auto      # or: dl / classical

Tesseract: Windows `winget install UB-Mannheim.TesseractOCR` (if not found, set
`pytesseract.pytesseract.tesseract_cmd` at the top of flow2drawio.py), Mac `brew install tesseract`,
Linux `sudo apt install tesseract-ocr`.

## Train on your own GPU (RTX 4050)
    # Windows: plain `pip install torch` gives a CPU-only build. Install the CUDA build first, e.g.
    #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    # and check:  python -c "import torch; print(torch.cuda.is_available())"   -> must print True
    python gen_data.py --out data --train 3000 --val 300 --test 100     # synthetic data + exact labels
    python train.py --epochs 60 --imgsz 960 --batch 8                   # lower --batch to 4 if out of memory
    python eval_pipeline.py --method both                               # classical vs deep learning
The result is saved to `models/shape_detector.pt` and picked up automatically.

## Results (measured)
Shipped model: YOLOv8n, pretrained start, trained 8 epochs on 300 synthetic charts at 512 px on a CPU (13 min).
Validation (100 unseen charts, 1130 shapes): precision 1.00, recall 1.00, mAP50 0.995, mAP50-95 0.988.

Held-out synthetic test charts (60), clean images:
| pipeline  | node F1 | label acc | edge F1 (directed) | edge F1 (connection) | charts fully correct |
|-----------|---------|-----------|--------------------|----------------------|----------------------|
| classical | 1.000   | 0.997     | 0.951              | 0.997                | 41/60                |
| deep learning | 1.000 | 0.997    | 0.951              | 0.997                | 41/60                |

Both pipelines also reproduce the two demo images exactly (`python test_ground_truth.py`).
Honest note: on rectangles and diamonds the deep-learning detector ties the classical rules. Its advantage is
extensibility: a new shape class (oval, parallelogram...) needs new generated data + retraining, not new hand-written geometry rules.

## Known limits
- Clean, printed-style charts only. Heavily degraded images mostly hurt OCR (label accuracy drops).
- A line touching more than two shapes (a fan-out from one point) is skipped with a message.
- Arrow direction is missed on roughly 5% of edges in the synthetic test set.
- Only rectangles and diamonds; no edge labels; edges are straight lines (draw.io re-routes them if you move shapes).

## Troubleshooting
- **Browser says ERR_CONNECTION_RESET / "Failed to load page":** the Python process crashed. Look at the terminal where
  you ran `python app.py` - the real error is printed there. `app.py` now loads PyTorch first and warms the model up at
  startup, so a crash happens at launch with a message instead of during an upload. If warm-up fails, pick
  "Classical only" in the page (needs no PyTorch), or reinstall torch (see the CUDA note above).
- **Isolate the problem from the terminal (no web page involved):**
      python flow2drawio.py flowchart2.png out.drawio --method classical
      python flow2drawio.py flowchart2.png out.drawio --method dl
  If `classical` works and `dl` crashes, the problem is the PyTorch install, not this project.
- **`tesseract is not installed` / not found:** install Tesseract OCR and set `pytesseract.pytesseract.tesseract_cmd`.
- **The shipped model** was trained on a CPU (8 epochs, 300 charts, 512 px). Retrain on your GPU for a stronger one.
