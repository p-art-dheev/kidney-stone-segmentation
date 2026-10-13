"""Read-only inventory of tracked project files and historical notebook claims."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    inventory = []
    notebook_rows = []
    for name in filter(None, files):
        path = ROOT / name
        if not path.is_file():
            continue
        inventory.append({'path': name, 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        if path.suffix != '.ipynb':
            continue
        nb = json.loads(path.read_text(encoding='utf-8'))
        for i, cell in enumerate(nb['cells'], 1):
            source = ''.join(cell.get('source', []))
            outputs = []
            for output in cell.get('outputs', []):
                if 'text' in output:
                    outputs.append(''.join(output['text']))
                if 'text/plain' in output.get('data', {}):
                    outputs.append(''.join(output['data']['text/plain']))
            notebook_rows.append({'notebook': name, 'cell': i, 'type': cell['cell_type'],
                                  'source': source, 'stored_text_outputs': outputs,
                                  'evidence_class': 'current reproducible wrapper; no executed corrected-run outputs'})
    out = ROOT / 'reports/compliance-20261013'
    out.mkdir(exist_ok=True)
    (out / 'repository_inventory.json').write_text(json.dumps(inventory, indent=2)+'\n', encoding='utf-8')
    (out / 'notebook_claim_inventory.json').write_text(json.dumps(notebook_rows, indent=2)+'\n', encoding='utf-8')
    categories = {
        'Published literature': 'Local IEEE paper, Table 3 p.9; externally reported 150-epoch scores.',
        'Verified canonical measurements': 'reports/experiments/baseline10fp32_final: four models, twenty folds; Attention export: five folds.',
        'Derived results tables': 'summary.json, comparison.json, docs/BASELINE_RESULTS.md and ATTENTION_RESULTS.md; independently recomputed from counts.',
        'Historical results': 'Stored notebook outputs, Results/*/*.png and twenty .keras archives; final-epoch, old mask/metric/split protocol.',
        'Exploratory results': 'baseline10amp_verified: mixed precision; TransUNet NaN failure; never mix with canonical float32.',
        'Incomplete experiments': 'Focal Tversky has zero completed outer folds; no legitimate comparative score. 150-epoch runs unavailable.',
        'Functionality only': 'CPU/GPU smoke and checkpoint round-trip logs; not CV or clinical performance.',
        'Unverified human facts': 'Group number, names, roll numbers and individual contributions absent; no pre-existing report or presentation found.'}
    text = '# Repository inventory and claim sources\n\n'
    text += f'Inventoried {len(inventory)} tracked files and {len(notebook_rows)} cells across three notebooks. '
    text += 'Complete sources and retained text outputs are in `reports/compliance-20261013/notebook_claim_inventory.json`. '
    text += 'File sizes and SHA-256 values are in `repository_inventory.json`. These hashes inventory the starting evidence, not new generated outputs.\n\n'
    for key, value in categories.items():
        text += f'## {key}\n\n{value}\n\n'
    text += '## Notebook review\n\n'
    for name in sorted({r['notebook'] for r in notebook_rows}):
        rows = [r for r in notebook_rows if r['notebook'] == name]
        text += f'- `{name}`: {len(rows)} cells, {sum(bool(r["stored_text_outputs"]) for r in rows)} cells with stored text outputs. '
        text += 'Current notebooks delegate to the shared pipeline and have no stored execution outputs; legacy notebook versions remain in Git history.\n'
    text += '\nThe corrected package removes independent metric maxima and outer-fold epoch selection, validates strict pairs, '
    text += 'uses decoded masks >127, retains incomplete batches, seeds runs, and separates inner checkpoint selection from outer evaluation. '
    text += 'Architectural equivalence to all source-paper implementations is not established.\n'
    (ROOT / 'docs/REPOSITORY_INVENTORY.md').write_text(text, encoding='utf-8')
    print(f'{len(inventory)} files; {len(notebook_rows)} cells inventoried')


if __name__ == '__main__':
    main()
