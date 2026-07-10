import json
from pathlib import Path
p = Path(r'notebook/ModelBuilding (2).ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        if 'best_model = ' in src or 'best_models["XGBoost"]' in src or 'random_search_rf.best_estimator_' in src:
            print('CELL', i)
            print(src)
            print('---')
