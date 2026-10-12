"""
Compare the deep-learning pipeline with the classical pipeline on held-out synthetic charts
(which have exact ground truth), plus the two real demo images.

    python eval_pipeline.py                 # both methods on data/images/test
    python eval_pipeline.py --method dl --limit 20
"""
import argparse, glob, json, os, re, tempfile, time
import cv2, numpy as np
import flow2drawio as f


def perturb(path, level=1):
    """Degrade an image: blur + noise + heavy JPEG compression (level 1 = moderate, 2 = severe)."""
    img = cv2.imread(path).astype(np.float32)
    sigma, noise, q = {1: (1.2, 10, 35), 2: (2.0, 22, 15)}[level]
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    rng = np.random.default_rng(abs(hash(path)) % (2**32))
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    out = os.path.join(tempfile.mkdtemp(), "p.jpg")
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return out

DEMOS = {  # name -> (n_nodes, directed edges, n_diamonds)
 "flowchart.png": (12, {("Start","Load Data"),("Load Data","Validate"),("Validate","Error Handler"),("Validate","Parse Input"),
   ("Parse Input","Check Rules"),("Check Rules","Auth Verify"),("Auth Verify","Transform"),("Transform","Queue Task"),
   ("Queue Task","Execute"),("Execute","Monitor"),("Monitor","Complete"),("Complete","Load Data")}, 0),
 "flowchart2.png": (12, {("Start","Input"),("Input","Check Req"),("Check Req","Collect Data"),("Check Req","Validate Data"),
   ("Collect Data","Transform"),("Validate Data","Store"),("Transform","Data OK?"),("Store","Data OK?"),
   ("Transform","Quality Pass?"),("Store","Archive?"),("Data OK?","Generate Report"),("Quality Pass?","Generate Report"),
   ("Archive?","Generate Report"),("Generate Report","Complete")}, 4)}


def norm(s):
    return re.sub(r"[^a-z0-9?]", "", s.lower())


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)


def score_synthetic(img, gt, method):
    t = time.time()
    f.convert(img, method); dt = time.time() - t
    L = f.convert.last
    pred = [(x - 4, y - 4, x + w + 4, y + h + 4) for x, y, w, h in L["bboxes"]]   # hole bbox -> outer stroke bbox
    # greedy IoU matching of predicted shapes to ground-truth nodes
    match, used = {}, set()
    for g in gt["nodes"]:
        best, bi = None, 0.5
        for j, p in enumerate(pred):
            if j in used: continue
            v = iou(g["bbox"], p)
            if v > bi: best, bi = j, v
        if best is not None:
            match[g["id"]] = best; used.add(best)
    tp_nodes = sum(1 for g in gt["nodes"] if g["id"] in match and L["kinds"][match[g["id"]]] == g["kind"])
    lab_ok = sum(1 for g in gt["nodes"] if g["id"] in match and norm(L["labels"][match[g["id"]]]) == norm(g["label"]))
    inv = {v: k for k, v in match.items()}
    pe, pu = set(), set()
    for s, d, st, en in L["edges"]:
        if s in inv and d in inv:
            a, b = inv[s], inv[d]
            if en and not st: pe.add((a, b))
            elif st and not en: pe.add((b, a))
            pu.add(frozenset((a, b)))
    ge = {(e["src"], e["dst"]) for e in gt["edges"]}; gu = {frozenset(e) for e in ge}
    return dict(n_gt=len(gt["nodes"]), n_pred=len(pred), tp_nodes=tp_nodes, lab_ok=lab_ok,
                e_gt=len(ge), e_pred=len(L["edges"]), e_tp_dir=len(pe & ge), e_tp_und=len(pu & gu), time=dt,
                perfect=(tp_nodes == len(gt["nodes"]) == len(pred) and pe == ge and len(L["edges"]) == len(ge)))


def f1(tp, p, g):
    pr = tp / p if p else 0; rc = tp / g if g else 0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--split", default="test")
    ap.add_argument("--method", default="both", choices=["both", "dl", "classical"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--perturb", type=int, default=0, choices=[0, 1, 2], help="degrade images: 1 moderate, 2 severe")
    a = ap.parse_args()
    methods = ["classical", "dl"] if a.method == "both" else [a.method]
    files = sorted(glob.glob(f"{a.data}/images/{a.split}/*.png"))
    if a.limit: files = files[:a.limit]
    print(f"{len(files)} held-out synthetic charts from '{a.split}' split" + (f"  [DEGRADED level {a.perturb}: blur + noise + JPEG]" if a.perturb else "") + "\n")
    print(f"{'method':10s} {'node F1':>8s} {'label acc':>10s} {'edge F1 (dir)':>14s} {'edge F1 (conn)':>15s} {'perfect charts':>15s} {'sec/img':>8s}")
    for m in methods:
        R = [score_synthetic(perturb(p, a.perturb) if a.perturb else p, json.load(open(p.replace("images", "gt").replace(".png", ".json"))), m) for p in files]
        S = lambda k: sum(r[k] for r in R)
        print(f"{m:10s} {f1(S('tp_nodes'), S('n_pred'), S('n_gt')):8.3f} {S('lab_ok')/S('n_gt'):10.3f} "
              f"{f1(S('e_tp_dir'), S('e_pred'), S('e_gt')):14.3f} {f1(S('e_tp_und'), S('e_pred'), S('e_gt')):15.3f} "
              f"{sum(r['perfect'] for r in R):>8d}/{len(R):<6d} {S('time')/len(R):8.2f}")
    print("\nYour two demo images (real ground truth):")
    for m in methods:
        for img, (nn, edges, nd) in DEMOS.items():
            if not os.path.exists(img): continue
            f.convert(img, m); L = f.convert.last
            got = {(L["labels"][s], L["labels"][d]) for s, d, st, en in L["edges"] if en and not st}
            ok = len(L["labels"]) == nn and L["kinds"].count("diamond") == nd and got == edges
            print(f"  {m:10s} {img:16s} nodes={len(L['labels'])}/{nn}  diamonds={L['kinds'].count('diamond')}/{nd}  edges+directions={'PASS' if got == edges else 'FAIL'}  -> {'PASS' if ok else 'FAIL'}")
