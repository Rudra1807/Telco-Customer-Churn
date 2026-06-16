import json, sys

NB = "notebooks/03_Model_Training.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

before = len(cells)

# Structure: 
#   0   = header markdown
#   1   = data loading code
#   2-9 = LR/RF cells (FIRST copy - KEEP)
#   10-17 = LR/RF cells (SECOND copy - REMOVE - duplicate from double-patch)
#   18+ = original XGBoost, SHAP, save cells

fixed = cells[0:10] + cells[18:]
nb["cells"] = fixed

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

sys.stderr.write(f"Done: {before} -> {len(fixed)} cells\n")
