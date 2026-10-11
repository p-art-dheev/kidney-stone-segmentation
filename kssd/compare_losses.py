"""Validate matched loss runs and summarize training-derived area groups."""
import argparse
import json
from pathlib import Path
import numpy as np
from kssd.metrics import metrics_from_counts
from kssd.reporting import METRICS,aggregate


def area_boundaries(manifest,fold):
    areas={s['id']:s['stone_area'] for s in manifest['samples']}
    training=fold['train']+fold['selection']
    return np.quantile([areas[name] for name in training],[1/3,2/3]).tolist()


def area_group(area,boundaries):
    return 'small' if area<=boundaries[0] else 'medium' if area<=boundaries[1] else 'large'


def summarize_rows(rows):
    counts={key:sum(row[key] for row in rows) for key in ['tp','fp','fn','tn']}
    return {'n_images':len(rows),'counts':counts,
            'micro':metrics_from_counts(**counts) if rows else None,
            'macro':{key:float(np.mean([r[key] for r in rows])) for key in METRICS} if rows else None}


def validate_pair(baseline,alternative,manifest,splits):
    """Aggregate enforces checkpoint/count validity before matching protocols."""
    summaries=[aggregate(root,manifest,splits) for root in [baseline,alternative]]
    configs=[summary['configuration'] for summary in summaries]
    if configs[0]['loss']!='dice' or configs[1]['loss']!='focal_tversky':
        raise ValueError('Expected Dice baseline and Focal Tversky alternative')
    if {k:v for k,v in configs[0].items() if k!='loss'}!={k:v for k,v in configs[1].items() if k!='loss'}:
        raise ValueError('Loss comparison has mismatched training conditions')
    for fold in splits['folds']:
        records=[json.loads((Path(root)/f"fold-{fold['fold']}"/'provenance.json').read_text()) for root in [baseline,alternative]]
        for record,config in zip(records,configs):
            if record['status']!='complete' or record['config']!=config or record['fold']!=fold:
                raise ValueError('Provenance does not match completed experiment')
            if record['dataset_fingerprint']!=splits['dataset_fingerprint']:
                raise ValueError('Source dataset differs')
            if not record.get('initial_weights_sha256'):
                raise ValueError('Missing initial weight hash')
            if record['environment'].get('deterministic_ops') is not True:
                raise ValueError('Deterministic execution not recorded')
        if records[0]['initial_weights_sha256']!=records[1]['initial_weights_sha256']:
            raise ValueError('Loss experiment initial weights differ')
        # Git/reporting-tool versions may differ, computational sources may not.
        for source in ['data.py','train.py','models.py','tf_metrics.py','augmentation.py',
                       'config.py','splits.py','metrics.py','reproducibility.py','losses.py']:
            hashes=[record['environment'].get('source_sha256',{}).get(source) for record in records]
            if not hashes[0] or hashes[0]!=hashes[1]:
                raise ValueError(f'Loss experiment computational source differs: {source}')
        for key in ['python','platform','versions','all_installed_packages','gpu_details','deterministic_ops','pythonhashseed']:
            if records[0]['environment'].get(key)!=records[1]['environment'].get(key):
                raise ValueError(f'Loss experiment environment differs: {key}')
    return summaries


def compare(baseline,alternative,manifest,splits):
    summaries=validate_pair(baseline,alternative,manifest,splits)
    folds=[];all_rows={'dice':[],'focal_tversky':[]}
    for fold in splits['folds']:
        boundaries=area_boundaries(manifest,fold)
        evaluations=[json.loads((Path(root)/f"fold-{fold['fold']}"/'evaluation.json').read_text()) for root in [baseline,alternative]]
        bins={};paired=[]
        maps=[{row['id']:row for row in evaluation['per_image']} for evaluation in evaluations]
        for name in sorted(maps[0]):
            a,b=maps[0][name],maps[1][name]
            group=area_group(a['stone_area'],boundaries)
            paired.append({'id':name,'area_group':group,'stone_area':a['stone_area'],
                           'dice_difference_focal_minus_baseline':b['hard_mask_dice']-a['hard_mask_dice']})
            for loss,row in zip(all_rows,[a,b]):
                all_rows[loss].append({**row,'area_group':group})
        for group in ['small','medium','large']:
            rows=[[row for row in mapping.values() if area_group(row['stone_area'],boundaries)==group] for mapping in maps]
            stats=[summarize_rows(part) for part in rows]
            bins[group]={'dice':stats[0],'focal_tversky':stats[1],
                         'macro_dice_difference':stats[1]['macro']['hard_mask_dice']-stats[0]['macro']['hard_mask_dice'] if rows[0] else None}
        folds.append({'fold':fold['fold'],'area_tercile_boundaries_pixels':boundaries,'bins':bins,
                      'paired_per_image':paired,
                      'micro_differences':{key:evaluations[1]['micro'][key]-evaluations[0]['micro'][key] for key in METRICS}})
    pooled={group:{loss:summarize_rows([row for row in rows if row['area_group']==group]) for loss,rows in all_rows.items()}
            for group in ['small','medium','large']}
    differences={key:[fold['micro_differences'][key] for fold in folds] for key in METRICS}
    return {'status':'complete','scope':'Matched measured loss comparison, not clinical stone-size categories',
            'architecture':summaries[0]['architecture'],'baseline':summaries[0],'alternative':summaries[1],
            'size_definition':'Per-fold outer-training (train + selection) mask-area terciles; <= lower small, <= upper medium, else large',
            'folds':folds,'pooled_size_groups':pooled,
            'paired_fold_mean_difference':{key:float(np.mean(values)) for key,values in differences.items()},
            'paired_fold_sample_std_difference':{key:float(np.std(values,ddof=1)) for key,values in differences.items()},
            'limitations':'Content groups are not known patients; short budget is preliminary; size groups use pixels, not physical dimensions. No significance claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True);parser.add_argument('--alternative',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--manifest',default='reports/dataset/manifest.json');parser.add_argument('--splits',default='reports/splits.json')
    args=parser.parse_args()
    result=compare(args.baseline,args.alternative,json.loads(Path(args.manifest).read_text()),json.loads(Path(args.splits).read_text()))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
