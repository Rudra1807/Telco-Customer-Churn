import json, sys
nb = json.load(open('notebooks/03_Model_Training.ipynb'))
cells = nb['cells']
print(f'Total cells: {len(cells)}', file=sys.stderr)
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    tag = ''.join(ch if ord(ch) < 128 else '?' for ch in src[:80]).replace('\n',' ')
    print(f'  Cell {i:02d} [{c["cell_type"]:8s}]: {tag}', file=sys.stderr)
