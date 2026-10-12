"""Synthetic flowchart dataset generator (YOLO format + ground-truth graph).

    python gen_data.py --out data_new --train 3000 --val 300 --test 100

Draws black-on-white flowcharts in the same style as the demo images
(rounded rectangles, diamonds, straight arrows, bold sans-serif labels) and
writes, for every image:
  images/<split>/<split>_NNNNN.png   the picture
  labels/<split>/<split>_NNNNN.txt   YOLO boxes  (class 0 = rect, 1 = diamond)
  gt/<split>/<split>_NNNNN.json      ground-truth graph (nodes, directed edges) used by eval_pipeline.py
NOTE: reconstructed to match the README / data layout after the original was overwritten; refuses to write
into a non-empty --out so it can never clobber an existing dataset.
"""
import os, json, random, argparse
from PIL import Image, ImageDraw, ImageFont

CLASSES = ["rect", "diamond"]
RECT_WORDS = ["Start", "Input", "Load Data", "Validate", "Parse Input", "Transform", "Check Rules", "Auth Verify",
              "Queue Task", "Execute", "Monitor", "Complete", "Error Handler", "Collect Data", "Validate Data",
              "Store", "Generate Report", "Process", "Review", "Send Email", "Fetch", "Cache", "Save", "Output",
              "Log Event", "Retry", "Notify", "Merge", "Split", "Clean Data", "Export", "Deploy", "Build", "Test",
              "End", "Read File", "Write File", "Compute", "Render", "Publish"]
DIAMOND_WORDS = ["Check Req", "Data OK?", "Quality Pass?", "Archive?", "Valid?", "Approved?", "Retry?", "Done?",
                 "Auth OK?", "Cache Hit?", "Error?", "Is Empty?", "More Data?", "Passed?"]


def find_font():
    cands = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
             "C:/Windows/Fonts/segoeuib.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
    try:
        import matplotlib
        cands.insert(0, os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"))
    except Exception:
        pass
    for c in cands:
        if os.path.exists(c):
            return c
    raise SystemExit("No bold TTF font found - edit find_font() in gen_data.py to point at one.")


def fit_font(fp, text, max_w, start=34):
    size = start
    while size > 12:
        f = ImageFont.truetype(fp, size)
        if f.getlength(text) <= max_w:
            return f
        size -= 2
    return ImageFont.truetype(fp, 12)


def clip(center, half, kind, dx, dy):
    """Where a ray leaving `center` in direction (dx,dy) crosses the shape outline."""
    a, b = half
    if kind == "rect":
        t = min(a / abs(dx) if dx else 1e9, b / abs(dy) if dy else 1e9)
    else:
        t = 1.0 / (abs(dx) / a + abs(dy) / b)
    return center[0] + t * dx, center[1] + t * dy


def make_chart(rng, fp):
    W, H = rng.choice([2000, 2200, 2400]), rng.choice([1300, 1400, 1600])
    cols, rows = rng.randint(3, 5), rng.randint(3, 4)
    mx, my = 150, 110
    cw, ch = (W - 2 * mx) / cols, (H - 2 * my) / rows
    lw = rng.choice([3, 4, 5])
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    rect_w, diam_names = rng.sample(RECT_WORDS, len(RECT_WORDS)), rng.sample(DIAMOND_WORDS, len(DIAMOND_WORDS))
    cells = {}
    p_fill, p_diamond = rng.uniform(0.6, 0.9), rng.uniform(0.15, 0.4)
    for r in range(rows):
        for c in range(cols):
            if rng.random() > p_fill:
                continue
            kind = "diamond" if rng.random() < p_diamond else "rect"
            w = min(400, cw * 0.62) * rng.uniform(0.9, 1.0)
            h = min(110, ch * 0.38) if kind == "rect" else min(190, ch * 0.55)
            label = (diam_names if kind == "diamond" else rect_w).pop()
            cells[(r, c)] = dict(kind=kind, cx=mx + cw * (c + 0.5), cy=my + ch * (r + 0.5), w=w, h=h, label=label)

    # edges between horizontally / vertically adjacent cells, random direction
    edges = []
    for (r, c) in cells:
        for nb in ((r, c + 1), (r + 1, c)):
            if nb in cells and rng.random() < 0.65:
                edges.append(((r, c), nb) if rng.random() < 0.5 else (nb, (r, c)))
    used = {k for e in edges for k in e}
    cells = {k: v for k, v in cells.items() if k in used}          # drop isolated shapes
    keys = sorted(cells)
    idx = {k: i for i, k in enumerate(keys)}

    nodes = []
    for k in keys:
        n = cells[k]
        x0, y0, x1, y1 = n["cx"] - n["w"] / 2, n["cy"] - n["h"] / 2, n["cx"] + n["w"] / 2, n["cy"] + n["h"] / 2
        if n["kind"] == "rect":
            d.rounded_rectangle([x0, y0, x1, y1], radius=8, outline="black", width=lw)
            font = fit_font(fp, n["label"], n["w"] * 0.8)
        else:
            pts = [(n["cx"], y0), (x1, n["cy"]), (n["cx"], y1), (x0, n["cy"])]
            d.line(pts + [pts[0]], fill="black", width=lw, joint="curve")
            font = fit_font(fp, n["label"], n["w"] * 0.5)
        d.text((n["cx"], n["cy"] + 2), n["label"], font=font, fill="black", anchor="mm")
        pad = lw / 2
        nodes.append(dict(label=n["label"], kind=n["kind"], bbox=[x0 - pad, y0 - pad, x1 + pad, y1 + pad]))

    out_edges = []
    for a, b in edges:
        A, B = cells[a], cells[b]
        dx, dy = B["cx"] - A["cx"], B["cy"] - A["cy"]
        p0 = clip((A["cx"], A["cy"]), (A["w"] / 2, A["h"] / 2), A["kind"], dx, dy)
        p1 = clip((B["cx"], B["cy"]), (B["w"] / 2, B["h"] / 2), B["kind"], -dx, -dy)
        d.line([p0, p1], fill="black", width=lw)
        L = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / L, dy / L
        al, aw = rng.uniform(16, 24), rng.uniform(6, 9)                # arrowhead length / half-width
        base = (p1[0] - ux * al, p1[1] - uy * al)
        d.polygon([p1, (base[0] - uy * aw, base[1] + ux * aw), (base[0] + uy * aw, base[1] - ux * aw)], fill="black")
        out_edges.append([idx[a], idx[b]])
    return img, nodes, out_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="write into a folder that already has files")
    a = ap.parse_args()
    if not a.force and os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f"'{a.out}' already has files; pick another --out (or pass --force to add into it).")
    rng, fp = random.Random(a.seed), find_font()
    counts = {"train": a.train, "val": a.val, "test": a.test}
    for split in counts:
        for sub in ("images", "labels", "gt"):
            os.makedirs(os.path.join(a.out, sub, split), exist_ok=True)
        for i in range(counts[split]):
            img, nodes, edges = make_chart(rng, fp)
            name = f"{split}_{i:05d}"
            img.save(f"{a.out}/images/{split}/{name}.png")
            W, H = img.size
            with open(f"{a.out}/labels/{split}/{name}.txt", "w") as f:
                for n in nodes:
                    x0, y0, x1, y1 = n["bbox"]
                    f.write(f"{CLASSES.index(n['kind'])} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}\n")
            gt = dict(width=W, height=H,
                      nodes=[dict(id=k, label=n["label"], kind=n["kind"], bbox=[round(v, 1) for v in n["bbox"]]) for k, n in enumerate(nodes)],
                      edges=[dict(src=u, dst=v) for u, v in edges])
            json.dump(gt, open(f"{a.out}/gt/{split}/{name}.json", "w"))
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")
    print(f"wrote {sum(counts.values())} charts to {a.out}/ (data.yaml ready)")


if __name__ == "__main__":
    main()
