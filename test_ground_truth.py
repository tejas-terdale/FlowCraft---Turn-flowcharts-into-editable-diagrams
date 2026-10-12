"""Checks the converter against the known-correct answer for both demo images."""
import flow2drawio as f

GT = {
 "flowchart.png": (12, {("Start","Load Data"),("Load Data","Validate"),("Validate","Error Handler"),("Validate","Parse Input"),
   ("Parse Input","Check Rules"),("Check Rules","Auth Verify"),("Auth Verify","Transform"),("Transform","Queue Task"),
   ("Queue Task","Execute"),("Execute","Monitor"),("Monitor","Complete"),("Complete","Load Data")}, 0),
 "flowchart2.png": (12, {("Start","Input"),("Input","Check Req"),("Check Req","Collect Data"),("Check Req","Validate Data"),
   ("Collect Data","Transform"),("Validate Data","Store"),("Transform","Data OK?"),("Store","Data OK?"),
   ("Transform","Quality Pass?"),("Store","Archive?"),("Data OK?","Generate Report"),("Quality Pass?","Generate Report"),
   ("Archive?","Generate Report"),("Generate Report","Complete")}, 4),
}
ok = True
for method, (img, (n_nodes, edges, n_diamonds)) in [(m, kv) for m in ('classical', 'dl') for kv in GT.items()]:
    f.convert(img, method); L = f.convert.last
    got = {(L["labels"][s], L["labels"][d]) for s, d, st, en in L["edges"] if en and not st}
    checks = {"node count": len(L["labels"]) == n_nodes, "diamond count": L["kinds"].count("diamond") == n_diamonds,
              "edges + directions": got == edges}
    for name, passed in checks.items():
        print(f"{method:9s} {img:16s} {name:20s} {'PASS' if passed else 'FAIL'}"); ok &= passed
    if got != edges: print("   missing:", edges - got, "extra:", got - edges)
print("ALL PASS" if ok else "SOME FAILED")
