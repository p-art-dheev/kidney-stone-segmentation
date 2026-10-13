"""Check packaged sources and measured tables; create page contact sheets.

Requires pypdf and Pillow. Does not modify experimental evidence.
"""
import hashlib
import json
import re
from pathlib import Path
from pypdf import PdfReader
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tmp/audit-20261013'


def sheets(prefix):
    pages=sorted(OUT.glob(prefix+'-*.png'))
    for start in range(0,len(pages),4):
        images=[Image.open(p).convert('RGB') for p in pages[start:start+4]]
        width=max(x.width for x in images);height=max(x.height for x in images)+30
        canvas=Image.new('RGB',(width*2,height*2),'#ddd')
        for offset,(path,im) in enumerate(zip(pages[start:start+4],images)):
            x=(offset%2)*width;y=(offset//2)*height
            canvas.paste(im,(x,y+30));ImageDraw.Draw(canvas).text((x+12,y+7),path.stem,fill='black')
        canvas.save(OUT/f'{prefix}-sheet-{start//4+1}.png')


def main():
    report=ROOT/'report/main.tex';slides=ROOT/'presentation/main.tex'
    data=json.loads((ROOT/'reports/compliance-20261013/deliverable_data.json').read_text())
    source=report.read_text();deck=slides.read_text()
    assert r'\documentclass[a4paper,fleqn]{cas-sc}' in source
    assert 'beamer}' in deck
    assert deck.count(r'\begin{frame}')==15
    for name in re.findall(r'\\input\{([^}]+)\}',source):assert (report.parent/name).is_file(),name
    for name in re.findall(r'\\panel\{([^}]+)\}',source):assert (report.parent/'figures'/name).is_file(),name
    for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',deck):assert (report.parent/'figures'/name).is_file(),name
    for a,s in data['summaries'].items():
        original=ROOT/('reports/experiments/attention10fp32_20261012/attention_unet/summary.json' if a=='attention_unet' else f'reports/experiments/baseline10fp32_final/{a}/summary.json')
        assert s==json.loads(original.read_text()),a
        assert len(s['fold_metrics'])==5
        assert sum(f['epochs_completed'] for f in s['fold_metrics'])==50
        for f in s['fold_metrics']:
            for value in f['micro'].values():assert f'{value:.6f}' in (ROOT/'report/tables/folds.tex').read_text()
    assert sum(f['epochs_completed'] for s in data['summaries'].values() for f in s['fold_metrics'])==250
    for name,expected in data['figure_hashes'].items():
        assert hashlib.sha256((ROOT/'report/figures'/name).read_bytes()).hexdigest()==expected,name
    pdfs={}
    for folder in ['report','presentation']:
        path=ROOT/folder/'main.pdf';reader=PdfReader(path)
        assert len(reader.pages)>0
        if folder=='presentation':assert len(reader.pages)==15
        text='\n'.join(p.extract_text() or '' for p in reader.pages)
        assert '250' in text,folder
        assert not re.search(r'\[\?\]',text),folder
        pdfs[folder]={'pages':len(reader.pages),'bytes':path.stat().st_size,
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    sheets('report-page');sheets('slide')
    result={'status':'verified','checks':['CAS source','Beamer source and 15 pages','all referenced inputs and figures exist',
             'generated data matches original five-fold records','all fold table scores match canonical rounding',
             '250 aggregate epochs across 25 runs',f'{len(data["figure_hashes"])} figure identities match','no unresolved citation token'],
            'pdfs':pdfs,'visual_review':'pending human-like inspection of rendered page contact sheets'}
    (ROOT/'reports/compliance-20261013/deliverable-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(pdfs))


if __name__=='__main__':main()
