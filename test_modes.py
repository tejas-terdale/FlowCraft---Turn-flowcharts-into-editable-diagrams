"""Proves what each mode really runs. Usage:  python test_modes.py [image.png]

It spies on the YOLO model and on the classical detector, and also sabotages YOLO on purpose to check that
"dl" never silently falls back to classical (while "auto" does, by design).
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch                      # noqa: F401  (import before cv2 - see app.py)
from ultralytics import YOLO
import flow2drawio as f

IMG = sys.argv[1] if len(sys.argv) > 1 else "flowchart2.png"
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")


calls = {"yolo": 0, "classical": 0}
orig_cls = f._classical_shapes
f._classical_shapes = lambda bw: (calls.__setitem__("classical", calls["classical"] + 1), orig_cls(bw))[1]

model = YOLO(f.MODEL_PATH)
f._MODEL["m"] = model
real_predict = model.predict
mutate = {"fn": lambda r: r}            # how to tamper with YOLO's output
raise_in_predict = {"on": False}


def spy_predict(*a, **k):
    calls["yolo"] += 1
    if raise_in_predict["on"]:
        raise RuntimeError("simulated YOLO crash")
    res = real_predict(*a, **k)
    return [mutate["fn"](res[0])]


model.predict = spy_predict


def run(method):
    calls.update(yolo=0, classical=0)
    out = f.convert(IMG, method)
    return out, dict(calls)


print(f"model file : {f.MODEL_PATH}")
print(f"cuda       : {torch.cuda.is_available()}   trained imgsz: {model.overrides.get('imgsz')}")

print("\n[1] classical mode")
(_, n, _), c = run("classical")
check("YOLO not called", c["yolo"] == 0, f"(yolo calls={c['yolo']})")
check("classical detector used", c["classical"] == 1 and f.convert.last["method"] == "classical")

print("\n[2] dl mode")
(_, n_dl, _), c = run("dl")
check("YOLO called", c["yolo"] == 1, f"(yolo calls={c['yolo']})")
check("classical shape detector NOT called", c["classical"] == 0)
check("reports 'dl'", f.convert.last["method"] == "dl")
check("same node count as classical", n_dl == n, f"({n_dl} vs {n})")

print("\n[3] YOLO returns only 3 boxes -> dl result must be exactly those 3 (proves YOLO decides what exists)")
mutate["fn"] = lambda r: (setattr(r, "boxes", r.boxes[:3]), r)[1]
(_, n3, _), c = run("dl")
check("dl gives 3 nodes", n3 == 3, f"(got {n3}; classical alone would give {n})")
(_, n3a, _), c = run("auto")
check("auto also trusts YOLO when it finds something", n3a == 3 and c["classical"] == 0)

print("\n[4] YOLO finds nothing")
mutate["fn"] = lambda r: (setattr(r, "boxes", r.boxes[:0]), r)[1]
(_, n0, _), c = run("dl")
check("dl returns 0 nodes, NO fallback", n0 == 0 and c["classical"] == 0, f"(nodes={n0}, classical calls={c['classical']})")
(_, na, _), c = run("auto")
check("auto falls back to classical", na == n and f.convert.last["method"] == "classical", f"(nodes={na})")
mutate["fn"] = lambda r: r

print("\n[5] YOLO crashes")
raise_in_predict["on"] = True
try:
    run("dl"); check("dl raises instead of falling back", False, "(it returned a result!)")
except RuntimeError as e:
    check("dl raises instead of falling back", True, f"({e})")
(_, nf, _), c = run("auto")
check("auto falls back to classical", nf == n and f.convert.last["method"] == "classical")
raise_in_predict["on"] = False

print("\n[6] weights file missing")
good = f.MODEL_PATH
f.MODEL_PATH = "does/not/exist.pt"
try:
    run("dl"); check("dl raises when weights are missing", False)
except RuntimeError as e:
    check("dl raises when weights are missing", True)
(_, nm, _), c = run("auto")
check("auto falls back to classical", nm == n and f.convert.last["method"] == "classical")
f.MODEL_PATH = good

print("\nALL PASS" if ok else "\nSOME FAILED")
