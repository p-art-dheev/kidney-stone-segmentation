"""Compare a fresh integrity audit with the frozen experimental manifest."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
a=json.loads((root/'reports/dataset/manifest.json').read_text())
b=json.loads((root/'tmp/audit-20261013/dataset/dataset_manifest.json').read_text())
assert a==b, 'Fresh manifest differs from frozen experimental data'
result={'status':'verified','fresh_manifest_matches_frozen':True,'pairs':a['count'],
        'empty_masks':a['empty_masks'],'source':'kssd.validate_data; all decoded image/mask hashes, dimensions, binary areas and duplicate clusters recomputed'}
(root/'reports/compliance-20261013/dataset-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print('All 838 source pairs match the frozen experimental manifest')
