import sys, re, math, cv2, numpy as np, pytesseract
from xml.sax.saxutils import escape


def _ocr(bw, shape_mask, bbox):
    """OCR the text inside one shape. Only ink inside the shape's own outline counts."""
    x, y, w, h = bbox
    inner_mask = cv2.erode(shape_mask, np.ones((7, 7), np.uint8))[y:y + h, x:x + w]
    ink = cv2.bitwise_and(bw[y:y + h, x:x + w], inner_mask)
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        return "?"
    text = 255 - ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # black text on white
    for target_h in (60, 40, 90):
        sc = target_h / max(1, text.shape[0])
        c = cv2.resize(text, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        c = cv2.copyMakeBorder(c, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        for psm in (7, 6, 8):
            t = pytesseract.image_to_string(c, config=f"--psm {psm}").strip().replace("\n", " ")
            t = re.sub(r"[^\w\s\-/&().,:?!]", "", t).strip()
            if t:
                return t
    return "?"


DEBUG = False


import os

MODEL_PATH = os.environ.get("FLOWCHART_MODEL",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "shape_detector.pt"))
KINDS = ["rect", "diamond"]
_MODEL = {}


KERNEL_SHAPE = cv2.MORPH_RECT      # shape of the erase footprint
DIAMOND_EXTRA = 2                  # extra erase pixels around diamonds (sharp tips need a bit more)


def _kernel(k):
    return cv2.getStructuringElement(KERNEL_SHAPE, (2 * k + 1, 2 * k + 1))


def _erase_radius(shape):
    """How far around a shape's inner outline to erase, so the stroke is gone but arrowheads stay."""
    return 6 + (DIAMOND_EXTRA if shape["kind"] == "diamond" else 0)


def _sort_shapes(shapes):
    shapes.sort(key=lambda s: (round(s["bbox"][1] / 60), s["bbox"][0]))
    return shapes


def _classical_shapes(bw):
    """Fallback detector: rectangles/diamonds = white holes with the right fill ratio / corner count."""
    H, W = bw.shape
    cnts, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []                                   # dicts: kind, bbox, mask
    for c, h in zip(cnts, hier[0]):
        if h[3] == -1:                            # must be a hole
            continue
        x, y, w, hh = cv2.boundingRect(c)
        if not (0.0005 * H * W < w * hh < 0.25 * H * W) or w < 40 or hh < 25:
            continue
        fill = cv2.contourArea(c) / (w * hh)
        verts = len(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True))
        if fill >= 0.9:
            kind = "rect"
        elif 0.4 <= fill <= 0.6 and verts == 4:
            kind = "diamond"
        else:
            continue                              # e.g. the empty space enclosed by a loop of arrows
        m = np.zeros_like(bw)
        cv2.drawContours(m, [c], -1, 255, -1)
        if cv2.countNonZero(cv2.bitwise_and(bw, m)) < 30:
            continue                              # no text inside => empty gap between lines, not a shape
        shapes.append({"kind": kind, "bbox": (x, y, w, hh), "mask": m})
    return _sort_shapes(shapes)


def dl_available():
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def _dl_shapes(img, bw, conf=0.4):
    """Primary detector: a YOLO network finds and classifies shapes; the classical outline
    (white hole inside the stroke) is then used only to snap each box to the exact contour."""
    if "m" not in _MODEL:
        from ultralytics import YOLO
        _MODEL["m"] = YOLO(MODEL_PATH)
    model = _MODEL["m"]
    imgsz = int(model.overrides.get("imgsz", 960) or 960)
    res = model.predict(img, imgsz=imgsz, conf=conf, iou=0.5, verbose=False)[0]
    cnts, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = [(c, cv2.boundingRect(c)) for c, h in zip(cnts, hier[0]) if h[3] != -1]
    shapes = []
    for (x0, y0, x1, y1), cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
        best, best_iou = None, 0.0
        for c, (x, y, w, h) in holes:
            ix0, iy0, ix1, iy1 = max(x0, x - 4), max(y0, y - 4), min(x1, x + w + 4), min(y1, y + h + 4)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            union = (x1 - x0) * (y1 - y0) + (w + 8) * (h + 8) - inter
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best, best_iou = (c, (x, y, w, h)), iou
        m = np.zeros_like(bw)
        if best is not None and best_iou >= 0.7:
            cv2.drawContours(m, [best[0]], -1, 255, -1)
            bbox = best[1]
        else:                                     # no clean outline found: use the network's box, inset by the stroke
            x, y, w, h = int(x0 + 4), int(y0 + 4), int(x1 - x0 - 8), int(y1 - y0 - 8)
            bbox = (x, y, w, h)
            if KINDS[cls] == "rect":
                cv2.rectangle(m, (x, y), (x + w, y + h), 255, -1)
            else:
                cv2.fillPoly(m, [np.array([(x + w // 2, y), (x + w, y + h // 2), (x + w // 2, y + h), (x, y + h // 2)])], 255)
        shapes.append({"kind": KINDS[cls], "bbox": bbox, "mask": m})
    return _sort_shapes(shapes)


def convert(src, method="auto"):
    """Image path -> (drawio_xml, n_nodes, n_edges).
    method: "auto" (deep-learning detector if its weights exist, else classical), "dl", or "classical"."""
    img = cv2.imread(src)
    if img is None:
        raise ValueError("Could not read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    H, W = bw.shape

    # 1. Shapes: YOLO detector (primary) or classical contour analysis (fallback)
    shapes, used = None, "classical"
    if method in ("auto", "dl"):
        if dl_available():
            try:
                shapes, used = _dl_shapes(img, bw), "dl"
                if not shapes and method == "auto":
                    print("DL detector found no shapes; falling back to classical")
                    shapes = None
            except Exception as e:
                if method == "dl":
                    raise
                print("DL detector failed (%s); falling back to classical" % e)
        elif method == "dl":
            raise RuntimeError("DL weights not found at " + MODEL_PATH)
    if shapes is None:
        shapes, used = _classical_shapes(bw), "classical"

    # 2. OCR each shape
    for s in shapes:
        s["label"] = _ocr(bw, s["mask"], s["bbox"])

    # 3. Erase shapes (+ their border) -> only lines remain -> each line component = one edge
    erase = np.zeros_like(bw)
    for s in shapes:
        k = _erase_radius(s)
        s["erase"] = cv2.dilate(s["mask"], _kernel(k))
        s["near"] = cv2.dilate(s["mask"], _kernel(k + 12))      # reach for "touching"
        erase |= s["erase"]
    lines = cv2.bitwise_and(bw, cv2.bitwise_not(erase))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)

    edge_dbg = []
    edges = []                                    # (src_idx, dst_idx, arrow_at_a, arrow_at_b)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 30:
            continue
        comp = (lab == i).astype(np.uint8)
        touching = [j for j, s in enumerate(shapes) if (comp & (s["near"] > 0)).any()]
        if len(touching) != 2:
            print("skipped a line touching", len(touching), "shapes")
            continue
        a, b = touching

        # 4. Arrowhead = the end where the line gets fat. Compare widths near each shape
        #    against the width of the plain middle of the line.
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        za = (shapes[a]["near"] > 0)
        zb = (shapes[b]["near"] > 0)
        far = (comp > 0) & ~cv2.dilate(shapes[a]["near"], np.ones((25, 25), np.uint8)).astype(bool) \
                         & ~cv2.dilate(shapes[b]["near"], np.ones((25, 25), np.uint8)).astype(bool)
        mid_w = np.median(dt[far & (dt > 0)]) if far.any() else None
        wa = dt[(comp > 0) & za].max() if ((comp > 0) & za).any() else 0
        wb = dt[(comp > 0) & zb].max() if ((comp > 0) & zb).any() else 0
        if mid_w is None:                         # very short edge: compare the two ends only
            arrow_a, arrow_b = wa > 1.4 * wb, wb > 1.4 * wa
        else:
            base = dt[far].max()
            arrow_a, arrow_b = wa > 1.3 * base, wb > 1.3 * base
        if DEBUG: print('  dbg', shapes[a]['label'], '|', shapes[b]['label'], 'wa=%.1f wb=%.1f base=%s' % (wa, wb, None if mid_w is None else round(float(dt[far].max()),1)))
        edge_dbg.append((a, b, float(wa), float(wb), float(dt[far].max()) if mid_w is not None else None))
        edges.append((a, b, bool(arrow_a), bool(arrow_b)))

    # 5. Write draw.io file (original layout kept)
    xml = ['<?xml version="1.0" encoding="UTF-8"?><mxfile host="app.diagrams.net"><diagram name="Page-1" id="p1">'
           f'<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
           f'fold="1" page="1" pageScale="1" pageWidth="{W + 200}" pageHeight="{H + 200}" math="0" shadow="0"><root>'
           '<mxCell id="0"/><mxCell id="1" parent="0"/>']
    for i, s in enumerate(shapes):
        x, y, w, h = s["bbox"]
        if s["kind"] == "diamond":     # hole is inset from the outer outline; grow it back (stroke centre ~2px out)
            px = 2 / math.sin(math.atan2(h, w))
            py = 2 / math.sin(math.atan2(w, h))
            style = "rhombus;whiteSpace=wrap;html=1;fontStyle=1;fontSize=18;strokeWidth=3;"
        else:
            px = py = 2
            style = "rounded=1;whiteSpace=wrap;html=1;fontStyle=1;fontSize=18;strokeWidth=3;"
        xml.append(f'<mxCell id="n{i}" value="{escape(s["label"])}" style="{style}" vertex="1" parent="1">'
                   f'<mxGeometry x="{x - px:.0f}" y="{y - py:.0f}" width="{w + 2 * px:.0f}" height="{h + 2 * py:.0f}" '
                   f'as="geometry"/></mxCell>')
    for k, (a, b, aa, ab) in enumerate(edges):
        if aa and not ab:                     # arrowhead at a  => b -> a
            src_i, dst_i, start, end = b, a, False, True
        elif ab and not aa:
            src_i, dst_i, start, end = a, b, False, True
        else:                                 # both ends, or none
            src_i, dst_i, start, end = a, b, aa, ab
        style = ("endArrow=%s;startArrow=%s;html=1;strokeWidth=3;"
                 % ("block" if end else "none", "block" if start else "none"))
        xml.append(f'<mxCell id="e{k}" style="{style}" edge="1" parent="1" source="n{src_i}" target="n{dst_i}">'
                   f'<mxGeometry relative="1" as="geometry"/></mxCell>')
        edges[k] = (src_i, dst_i, start, end)
    xml.append('</root></mxGraphModel></diagram></mxfile>')
    convert.last = {"edge_dbg": edge_dbg, "method": used, "bboxes": [s["bbox"] for s in shapes], "labels": [s["label"] for s in shapes], "kinds": [s["kind"] for s in shapes], "edges": edges}
    return "".join(xml), len(shapes), len(edges)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out", nargs="?", default="diagram.drawio")
    ap.add_argument("--method", choices=["auto", "dl", "classical"], default="auto")
    args = ap.parse_args()
    out = args.out
    xml, n, m = convert(args.image, args.method)
    open(out, "w", encoding="utf-8").write(xml)
    print(f"{n} nodes, {m} edges -> {out}  (detector: {convert.last['method']})")
    L = convert.last
    for i, (l, k) in enumerate(zip(L["labels"], L["kinds"])):
        print(f"  [{i}] {k:7s} {l}")
    for s, d, st, en in L["edges"]:
        print(f"  {L['labels'][s]} {'<' if st else '-'}-{'>' if en else '-'} {L['labels'][d]}")
