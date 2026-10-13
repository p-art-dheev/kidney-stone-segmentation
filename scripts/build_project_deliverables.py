"""Generate presentation-ready tables and figures exclusively from audited records.

Run in the documented training runtime; never starts training or inference.
"""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from kssd.data import read_grayscale
from kssd.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'report'
FIGS = REPORT / 'figures'
TABLES = REPORT / 'tables'
BASE = ROOT / 'reports/experiments/baseline10fp32_final'
ATT = ROOT / 'reports/experiments/attention10fp32_20261012'
NAMES = {'modified_unet':'Modified U-Net','unetpp':'U-Net++','unet3plus':'U-Net3+',
         'transunet':'TransUNet','attention_unet':'Attention U-Net'}
KEYS = ['hard_mask_dice','hard_mask_iou','hard_mask_precision','hard_mask_recall']
# Visually transcribed from Bouzon et al. Table 3 (primary PDF p.9); percentages.
PAPER = {'modified_unet':[(97.06,.39),(94.65,.62),(97.83,.33),(96.61,.77)],
         'unetpp':[(96.63,.63),(93.97,.83),(97.16,.68),(96.44,.67)],
         'unet3plus':[(96.68,.61),(94.02,.86),(96.88,.34),(96.90,1.02)],
         'transunet':[(95.53,.89),(92.21,1.27),(96.26,.52),(95.39,1.24)]}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text+'\n',encoding='utf-8')


def escape(text):
    return str(text).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def table(filename, caption, headers, rows, label=None):
    # Longtables retain headers across pages and avoid shrinking dense fold rows.
    cols=(r'p{0.34\linewidth}p{0.60\linewidth}' if len(headers)==2 else 'l'+'r'*(len(headers)-1))
    text=r'\small'+'\n'+r'\begin{longtable}{'+cols+'}\n'
    text+=r'\caption{'+caption+'}'+(r'\label{tab:'+label+'}' if label else '')+r'\\'+'\n'
    text+=r'\toprule'+'\n'+' & '.join(headers)+r'\\\midrule'+'\n'+r'\endfirsthead'+'\n'
    text+=r'\toprule'+'\n'+' & '.join(headers)+r'\\\midrule'+'\n'+r'\endhead'+'\n'
    text+='\n'.join(' & '.join(str(x) for x in row)+r'\\' for row in rows)
    text+='\n'+r'\bottomrule\end{longtable}\normalsize'
    write(TABLES/filename,text)


def mdtable(headers, rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+ ''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)


def pm(s,k,latex=False):
    a,b=s['fold_mean'][k],s['fold_sample_std'][k]
    return f'${a:.4f}\\pm{b:.4f}$' if latex else f'{a:.4f} ± {b:.4f}'


def node(ax,x,y,text,w=2.05,h=.65,color='#e7eef5',size=10):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.04',fc=color,ec='#47647d',lw=1))
    ax.text(x,y,text,ha='center',va='center',fontsize=size)


def arrow(ax,x,y,a,b,color='#426078'):
    ax.annotate('',xy=(a,b),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=color,lw=1.3))


def architecture(attention=False):
    fig,ax=plt.subplots(figsize=(12,6));ax.set(xlim=(-1.5,10.8),ylim=(-.9,5.4));ax.axis('off')
    for level,(res,ch) in enumerate(zip([512,256,128,64],[16,32,64,128])):
        y=4.4-level
        node(ax,1,y,f'{res} x {res} x {ch}\nConv / BN / ReLU',size=10)
        node(ax,8,y,f'{res} x {res} x {ch}\nConcat + conv block',size=10)
        if attention:
            node(ax,4.5,y,f'Gate {level+1}\n1x1 projections',w=1.8,color='#f4e2c7',size=9)
            arrow(ax,2.08,y,3.52,y);arrow(ax,5.48,y,6.9,y)
            # Context is the upsampled decoder stream before concatenation,
            # not the output of the concatenation block (which would be a cycle).
            arrow(ax,8,y-.48,5.3,y-.1,color='#bb791d')
            ax.plot(8,y-.48,'o',color='#bb791d',markersize=3)
            ax.text(8.25,y-.48,'upsampled g',fontsize=8,color='#8e5613',va='center')
        else:arrow(ax,2.08,y,6.9,y)
        if level<3:
            arrow(ax,1,y-.35,1,y-.65);arrow(ax,8,y-.65,8,y-.35)
    node(ax,4.5,.4,'32 x 32 x 256\nBottleneck',w=2.1)
    arrow(ax,1,1.05,3.35,.4);arrow(ax,5.65,.4,8,1.05)
    ax.text(1,5.1,'Input 512 x 512 x 1',ha='center',fontsize=11)
    ax.text(8,5.1,'1x1 sigmoid -> 512 x 512 x 1',ha='center',fontsize=11)
    ax.text(1,-.22,'Encoder: 2x 3x3 bias-free conv\nBN + ReLU; dropout; max pool',ha='center',fontsize=10)
    ax.text(8,-.22,'Decoder: 2x2 transposed conv\nstride 2; concatenate; 2x conv',ha='center',fontsize=10)
    if attention:ax.text(4.5,-.55,'a = sigmoid(psi(ReLU(Ws s + Wg g))); gated skip = a * s',ha='center',fontsize=10,color='#8e5613')
    fig.tight_layout();fig.savefig(FIGS/('attention-architecture.png' if attention else 'baseline-architecture.png'),dpi=220);plt.close(fig)


def variants():
    fig,axes=plt.subplots(1,3,figsize=(13,5))
    labels=[('U-Net++','Nested skip convolution nodes\nDense pathways between levels','No deep supervision\nNo pretrained backbone'),
            ('U-Net3+','Pool / upsample full-scale features\nProject 22 channels; fuse 110','No classification-guided module\nNo hybrid loss / deep supervision'),
            ('TransUNet','32x32 grid -> 1024 tokens\n192-d embedding; 3 blocks\n4 heads; MLP 384 / 192','No pretrained backbone\nBottleneck patch size 1')]
    for ax,(title,middle,footer) in zip(axes,labels):
        ax.set(xlim=(-1,4),ylim=(-.8,4.5));ax.axis('off');ax.set_title(title,fontsize=14)
        node(ax,1.5,3.8,'512x512x1 -> CNN encoder',w=4,size=10)
        node(ax,1.5,2.35,middle,w=4,h=1.2,color='#f4e2c7',size=10)
        node(ax,1.5,.9,'Decoder -> 1-channel sigmoid',w=4,size=10)
        arrow(ax,1.5,3.45,1.5,3);arrow(ax,1.5,1.7,1.5,1.28)
        ax.text(1.5,-.05,footer,ha='center',va='top',fontsize=10)
    fig.tight_layout();fig.savefig(FIGS/'variant-architectures.png',dpi=220);plt.close(fig)


def mask_for(case):
    path=ATT/'attention_unet'/f'fold-{case["fold"]}'/'predictions'/f'{Path(case["id"]).stem}.png'
    if not path.exists():
        # The exported prediction path is discovered from observed files, not invented.
        candidates=list(ATT.rglob(f'{Path(case["id"]).stem}.png'))
        if len(candidates)!=1:raise ValueError(f'Prediction path ambiguity: {case["id"]}: {candidates}')
        path=candidates[0]
    return np.asarray(Image.open(path))>0


def examples(index):
    chosen=[next(c for c in index['cases'] if c['criterion']==k) for k in ['highest_dice','middle_dice','largest_fn']]
    fig,axes=plt.subplots(3,6,figsize=(14,9))
    titles=['Input CT','Ground truth','Predicted mask','Input crop','GT crop','Prediction crop']
    for r,c in enumerate(chosen):
        x=read_grayscale(ROOT/'data/kssd2025/data/image'/c['id'])
        gt=read_grayscale(ROOT/'data/kssd2025/data/label'/c['id'])>127
        pred=mask_for(c);scores=binary_metrics(gt,pred.astype(float))
        for k in ['tp','fp','fn','tn',*KEYS]:
            if not np.isclose(scores[k],c[k]):raise ValueError(f'Case score mismatch: {c["id"]}/{k}')
        ys,xs=np.where(gt);cy,cx=int(np.median(ys)),int(np.median(xs));half=30
        sy=slice(max(0,cy-half),min(512,cy+half));sx=slice(max(0,cx-half),min(512,cx+half))
        for col,arr in enumerate([x,gt,pred,x[sy,sx],gt[sy,sx],pred[sy,sx]]):
            axes[r,col].imshow(arr,cmap='gray',interpolation='nearest',vmin=0,vmax=255 if col in [0,3] else 1)
            axes[r,col].set_xticks([]);axes[r,col].set_yticks([])
            if r==0:axes[r,col].set_title(titles[col],fontsize=11)
        axes[r,0].set_ylabel(f'{c["id"]}\nFold {c["fold"]}\nDice {c["hard_mask_dice"]:.4f}\nIoU {c["hard_mask_iou"]:.4f}',fontsize=10)
    fig.tight_layout();fig.savefig(FIGS/'required-three-examples.png',dpi=220);plt.close(fig)
    return chosen


def logs():
    rows=list(csv.DictReader((BASE/'modified_unet/fold-1/history.csv').open()))
    fig,ax=plt.subplots(figsize=(11,3.6));ax.axis('off')
    ax.text(.01,.95,'Original CSV log: Modified U-Net, fold 1',fontsize=15,weight='bold',transform=ax.transAxes)
    header='Epoch    Train loss    Inner loss    Train Dice    Inner Dice'
    lines=[header]+[f'{int(r["epoch"])+1:>5}    {float(r["loss"]):>10.6f}    {float(r["val_loss"]):>10.6f}    {float(r["hard_mask_dice"]):>10.6f}    {float(r["val_hard_mask_dice"]):>10.6f}' for r in rows]
    ax.text(.01,.83,'\n'.join(lines),va='top',family='monospace',fontsize=11,transform=ax.transAxes)
    fig.tight_layout();fig.savefig(FIGS/'training-log.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,2.7));ax.axis('off')
    ax.text(.02,.96,'Baseline fold 1: original log',fontsize=15,weight='bold',va='top')
    lines=['Epoch   Loss    Inner Dice']+[f'{int(rows[i]["epoch"])+1:>5}   {float(rows[i]["loss"]):.4f}    {float(rows[i]["val_hard_mask_dice"]):.4f}' for i in [0,4,8,9]]
    ax.text(.02,.72,'\n'.join(lines),va='top',family='monospace',fontsize=15)
    ax.text(.02,.02,'Selected epoch: 9 (inner maximum)',fontsize=13)
    fig.tight_layout();fig.savefig(FIGS/'slide-training-log.png',dpi=220);plt.close(fig)
    h=read(BASE/'modified_unet/fold-1/history.json')
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    for ax,key,title in zip(axes,['loss','hard_mask_dice'],['Loss','Hard Dice']):
        ax.plot(range(1,11),h[key],label='Training',lw=2)
        ax.plot(range(1,11),h['val_'+key],label='Inner selection',ls='--',lw=2)
        ax.axvline(9,color='#888',ls=':',lw=1)
        ax.set_xlabel('Epoch',fontsize=15);ax.set_title(title,fontsize=17);ax.tick_params(labelsize=12);ax.grid(alpha=.2)
    axes[1].legend(fontsize=11);fig.suptitle('Modified U-Net: fold 1; selected epoch 9',fontsize=15)
    fig.tight_layout();fig.savefig(FIGS/'slide-baseline-curves.png',dpi=220);plt.close(fig)


def slide_examples(chosen):
    # Preserve full required views; magnification is an inset, not a substitute.
    fig,axes=plt.subplots(3,3,figsize=(10,6.4))
    for r,c in enumerate(chosen):
        x=read_grayscale(ROOT/'data/kssd2025/data/image'/c['id'])
        gt=read_grayscale(ROOT/'data/kssd2025/data/label'/c['id'])>127;pred=mask_for(c)
        ys,xs=np.where(gt);cy,cx=int(np.median(ys)),int(np.median(xs));half=30
        sy=slice(max(0,cy-half),min(512,cy+half));sx=slice(max(0,cx-half),min(512,cx+half))
        for col,arr in enumerate([x,gt,pred]):
            ax=axes[r,col];ax.imshow(arr,cmap='gray',vmin=0,vmax=255 if col==0 else 1,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if r==0:ax.set_title(['Input CT','Ground truth','Predicted mask'][col],fontsize=18)
            inset=ax.inset_axes([.63,.63,.35,.35]);inset.imshow(arr[sy,sx],cmap='gray',vmin=0,vmax=255 if col==0 else 1,interpolation='nearest');inset.set_xticks([]);inset.set_yticks([])
            for spine in inset.spines.values():spine.set_edgecolor('#f0b545');spine.set_linewidth(1.5)
        axes[r,0].set_ylabel(c['id']+'\nDice '+f'{c[KEYS[0]]:.4f}',fontsize=16)
    fig.tight_layout();fig.savefig(FIGS/'slide-three-examples.png',dpi=240);plt.close(fig)


def main():
    FIGS.mkdir(parents=True,exist_ok=True);TABLES.mkdir(exist_ok=True)
    manifest=read(ROOT/'reports/dataset/manifest.json');splits=read(ROOT/'reports/splits.json')
    summaries={a:read(BASE/a/'summary.json') for a in list(NAMES)[:4]}
    summaries['attention_unet']=read(ATT/'attention_unet/summary.json')
    config=summaries['modified_unet']['configuration']
    for a,s in summaries.items():
        if any(s['configuration'][k]!=v for k,v in config.items() if k!='architecture'):
            raise ValueError(f'Mismatched comparison config: {a}')
    dataset_rows=[['Source image-mask pairs',manifest['count']],['Eligible unique images',sum(s['eligible'] for s in manifest['samples'])],
                  ['Excluded exact duplicates',7],['Image / mask shape','512 x 512'],['Formats','TIFF / uint8 grayscale'],
                  ['Labels','0 background; 1 stone'],['Empty masks',manifest['empty_masks']],['Masks with area 1-10 pixels',manifest['small_masks_le10_pixels']],
                  ['Area min / Q1 / median / Q3 / max','3 / 38.25 / 101.5 / 231 / 755'],['Content groups / largest original cluster','39 / 173'],
                  ['Normalization / mask rule','uint8 / 255; decoded mask >127']]
    table('dataset.tex','Audited original dataset statistics; areas are pixels, not physical sizes.',['Item','Value'],[[escape(x) for x in r] for r in dataset_rows])
    split_rows=[[f['fold'],f['seed'],len(f['train']),len(f['selection']),len(f['test'])] for f in splits['folds']]
    table('splits.tex','Frozen content-group fold sizes. No independent external test cohort.',['Fold','Seed','Training','Inner selection','Outer test'],split_rows)
    parameters=[[NAMES[a],s['fold_metrics'][0]['parameters']] for a,s in summaries.items()]
    table('parameters.tex','Total model parameters measured from the implemented models.',['Architecture','Parameters'],parameters)
    config_rows=[[NAMES[a],10,2,'float32','Adam','0.001','Dice'] for a in NAMES]
    table('configuration.tex','Matched executed settings. All models use seed 42 with fold seeds 43-47, augmentation and prediction threshold 0.5.',['Architecture','Epochs','Batch','Precision','Optimizer','LR','Loss'],config_rows)
    files_rows=[['Data and augmentation','kssd/data.py; augmentation.py'],['Architectures / losses','kssd/models.py; tf_metrics.py; losses.py'],['Training / evaluation','kssd/train.py'],['Integrity / splits','validate_data.py; splits.py'],['Aggregation / auditing','reporting.py; final_audit.py; attention_audit.py'],['Selected-checkpoint records','reports/experiments/*/architecture/fold-N/'],['Tests / runtime','tests/; reports/resources/wsl_gpu.lock.txt'],['Historical assets','Results/; three root notebooks'],['Current deliverables','report/; presentation/; docs/']]
    table('files.tex','Repository implementation and evidence map. Paths abbreviated after kssd/ where appropriate.',['Purpose','File or directory'],[[escape(x) for x in r] for r in files_rows])
    paper_rows=[];paper_md=[]
    for a in PAPER:
        s=summaries[a];p=PAPER[a]
        paper_rows.append([NAMES[a],f'${p[0][0]/100:.4f}\\pm{p[0][1]/100:.4f}$',pm(s,KEYS[0],True),f'${p[1][0]/100:.4f}\\pm{p[1][1]/100:.4f}$',pm(s,KEYS[1],True)])
        paper_md.append([NAMES[a],f'{p[0][0]/100:.4f} ± {p[0][1]/100:.4f}',pm(s,KEYS[0]),f'{p[1][0]/100:.4f} ± {p[1][1]/100:.4f}',pm(s,KEYS[1])])
    table('paper.tex','Literature versus our measured overlap results (Bouzon et al. Table 3 and canonical records). Different protocols; settings are documented separately.',['Architecture','Paper Dice','Local Dice','Paper IoU','Local IoU'],paper_rows)
    aggregate=[[NAMES[a]]+[pm(s,k,True) for k in KEYS] for a,s in summaries.items()]
    aggregate_md=[[NAMES[a]]+[pm(s,k) for k in KEYS] for a,s in summaries.items()]
    table('aggregate.tex','Measured unweighted fold mean and sample SD (ddof=1) of pooled-pixel outer metrics.',['Architecture','Dice','IoU','Precision','Recall'],aggregate)
    folds=[];fold_md=[];counts=[];runtime=[]
    for a,s in summaries.items():
        durations=[]
        for f in s['fold_metrics']:
            folds.append([NAMES[a],f['fold'],f['selected_epoch']]+[f'{f["micro"][k]:.6f}' for k in KEYS])
            fold_md.append(folds[-1]+[f'{f["training_seconds"]:.2f}'])
            counts.append([NAMES[a],f['fold']]+[f['counts'][k] for k in ['tp','fp','fn','tn']])
            folder=(ATT/'attention_unet' if a=='attention_unet' else BASE/a)/f'fold-{f["fold"]}'
            durations.append(read(folder/'evaluation.json')['inference_seconds_per_image_excluding_first_batch'])
        runtime.append([NAMES[a],f'{s["total_training_seconds"]:.2f}',f'{min(durations)*1000:.3f}--{max(durations)*1000:.3f}'])
    table('folds.tex','All selected-checkpoint outer-fold scores; zero-overlap folds retained.',['Architecture','Fold','Epoch','Dice','IoU','Precision','Recall'],folds)
    table('counts.tex','Verified pooled integer confusion counts per outer fold; each architecture covers 217,841,664 pixels.',['Architecture','Fold','TP','FP','FN','TN'],counts)
    table('runtime.tex','Measured summed training duration and range of per-fold inference milliseconds per image. Forward and synchronization only; first batch excluded.',['Architecture','Training seconds','Inference ms/image'],runtime)
    table('pooled.tex','All-fold pixel-pooled scores differ from the unweighted fold mean in the main comparison.',['Architecture','Pooled Dice','Pooled IoU','Pooled precision','Pooled recall'],[[NAMES[a]]+[f'{s["pooled_micro"][k]:.6f}' for k in KEYS] for a,s in summaries.items()],label='pooled')
    mod_rows=[[NAMES[a]]+[pm(summaries[a],k,True) for k in KEYS] for a in ['modified_unet','attention_unet']]
    deltas=[summaries['attention_unet']['fold_mean'][k]-summaries['modified_unet']['fold_mean'][k] for k in KEYS]
    mod_rows.append(['Attention minus baseline']+[f'{d:+.6f}' for d in deltas])
    table('modification.tex','Architecture-only controlled comparison. Differences are descriptive, not significance estimates.',['Model / difference','Dice','IoU','Precision','Recall'],mod_rows)
    # Reuse verified source figure bytes; no duplicate presentation figure collection.
    for src in (ROOT/'reports/figures/baseline10fp32_final').glob('*.png'):shutil.copyfile(src,FIGS/src.name)
    for src in (ATT/'figures').glob('*.png'):shutil.copyfile(src,FIGS/('attention-comparison.png' if src.name=='comparison.png' else src.name))
    architecture();architecture(True);variants();logs()
    chosen=examples(read(ATT/'figures/index.json'))
    slide_examples(chosen)
    table('examples.tex','Exact counts for the three mandatory-view examples; masks are original saved Attention predictions.',['Image','Fold','TP','FP','FN','Dice','IoU'],[[c['id'],c['fold'],c['tp'],c['fp'],c['fn'],f'{c[KEYS[0]]:.6f}',f'{c[KEYS[1]]:.6f}'] for c in chosen])
    # Pair existing full-view panels for the shared minimum-area sample.
    left=Image.open(FIGS/'modified_unet-smallest_area-126.png').convert('RGB');right=Image.open(FIGS/'attention_unet-smallest_area-126.png').convert('RGB')
    width=max(left.width,right.width);canvas=Image.new('RGB',(width,left.height+right.height),'white')
    canvas.paste(left,(0,0));canvas.paste(right,(0,left.height));canvas.save(FIGS/'paired-small-stone.png')
    all_tables='\n\n'.join(['## A. Dataset statistics\n\n'+mdtable(['Item','Value'],dataset_rows),
       '### Frozen split sizes\n\n'+mdtable(['Fold','Seed','Train','Inner','Outer'],split_rows),
       '## B. Training configurations\n\n'+mdtable(['Architecture','Epochs','Batch','Precision','Optimizer','LR','Loss'],config_rows),
       'Total workload: 250 training epochs across 25 fold runs. All: base seed 42, fold seeds 43-47, train-only paired augmentation, hard threshold >=0.5, earliest maximum inner hard Dice checkpoint. RTX 4070 Laptop; TF 2.20 / Keras 3.13.2; full lock in reports/resources/wsl_gpu.lock.txt. Training protocols differ from the source paper; see report settings section.',
       '## C. Base paper versus local replication attempt\n\n'+mdtable(['Architecture','Paper Dice','Local Dice','Paper IoU','Local IoU'],paper_md),
       'Results-only comparison: paper Table 3 (PDF p.9) versus canonical local records. Literature and local protocols differ; training settings are documented separately.',
       '## D. Model comparison\n\n'+mdtable(['Architecture','Dice','IoU','Precision','Recall'],aggregate_md),
       'Mean ± sample SD of five fold-pooled metrics, not confidence intervals or per-image averages.\n\n'+mdtable(['Architecture','Fold','Selected epoch','Dice','IoU','Precision','Recall','Training s'],fold_md),
       '### Parameters and runtime\n\n'+mdtable(['Architecture','Parameters'],parameters)+'\n'+mdtable(['Architecture','Training s','Inference ms/image range'],runtime),
       '## E. Baseline versus modification\n\n'+mdtable(['Comparison','Dice','IoU','Precision','Recall'],[[NAMES[a]]+[pm(summaries[a],k) for k in KEYS] for a in ['modified_unet','attention_unet']]+[['Attention minus baseline']+[f'{d:+.6f}' for d in deltas]]),
       'Modification: four additive spatial skip gates, 22,124 additional parameters. Config differs only in architecture; same frozen splits/seeds/budget/selection. Negative mean result; no improvement claim, no conditional improvement ablation. Focal Tversky and 150-epoch scores: NOT YET MEASURED.'])
    write(ROOT/'docs/EXPERIMENTAL_RESULTS.md','# Verified experimental results\n\nGenerated from canonical JSON; source map: report tables and reports/experiments.\n\n'+all_tables)
    slides=[
      ('Problem, motivation and team','Pixel segmentation localizes annotated kidney stones on 2D exported CT. Manual outlining motivates automation (Bouzon et al.). No clinical deployment claim. Put group ID, every name and roll number here.','report/main.tex Introduction; references.bib bouzon2025','baseline-architecture.png','Names, roll numbers, group ID.','Why masks, rather than image-level labels? Masks describe pixel location and extent.'),
      ('Dataset, objectives and metrics','838 matched pairs; seven redundant copies excluded -> 831 eligible. 512x512 grayscale; binary stone/background; no external project test cohort. Fold outer sizes 179/279/118/172/83. Dice and IoU primary, precision/recall secondary.','Tables A and split sizes; report Dataset and Evaluation','required-three-examples.png','None beyond roster.','Inner group selection chooses checkpoint; outer folds estimate held-out performance. Groups are not patient IDs.'),
      ('Base paper and literature','Use full eight-author citation, complete title, IEEE Access 13 (2025), 171790-171806; DOI 10.1109/ACCESS.2025.3610027. Dataset plus four lightweight architectures.','report/main.tex Base paper; paper/ primary PDF p.1','baseline-architecture.png','None.','The paper authors created KSSD2025 and their lightweight Modified U-Net; do not claim those as our inventions.'),
      ('Baseline architecture','Encoder channels 16/32/64/128; bottleneck 256; two 3x3 conv + BN + ReLU per block. Encoder dropout .1/.2/.3/.3, bottleneck .3. Max pooling and equal-resolution skips.','kssd/models.py; primary paper Fig.5 PDF p.7','baseline-architecture.png','None.','Explain spatial compression and skip recovery; diagram is attributed redrawing with code-specific details.'),
      ('Decoder and implemented alternatives','Transposed-conv baseline decoder and 1x1 sigmoid. U-Net++ nested skips; U-Net3+ full-scale fusion (22 projection, 110 aggregate); TransUNet 1024 tokens, 192 embedding, 3 blocks/4 heads. No deep supervision or pretrained backbone.','report Methodology; tables/parameters.tex','variant-architectures.png','None.','Describe only components in local code, not all enhancements in source papers.'),
      ('Published versus local results','Use results-only table C. Paper U-Net Dice .9706/IoU .9465; local baseline .7910/.6589. Preserve all four rows and standard deviations. Protocols differ; settings are documented separately.','Table C; primary paper Table3 PDF p.9; canonical summary.json','comparison.png','None.','Several protocol changes coexist; the result table does not establish an exact replication.'),
      ('Three required segmentation examples','Display 128.tif, 493.tif, 1068.tif as full Input / Ground Truth / Predicted Mask plus crops. Attention TP/FP/FN 21/0/0, 20/4/1, 0/4/629; Dice 1/.8889/0; IoU 1/.8/0.','report Qualitative; tables/examples.tex; figure index','required-three-examples.png','None.','Highest/middle/largest-FN selection is deliberate descriptive error analysis, not an unbiased sample.'),
      ('Implementation and reproducibility','data.py -> augmentation.py -> models.py -> train.py -> reporting.py/auditors. Frozen manifest/splits; deterministic new optimizer each fold; preserve remainder batches; earliest inner maximum.','report Implementation; docs/REPRODUCIBILITY.md; kssd/','training-log.png','Plan who runs the live demonstration.','Demo a records-only auditor and explain what it verifies; inference requires data and local weights.'),
      ('Training settings, logs and curves','Table B: TF2.20/Keras3.13.2; Adam .001; batch2; 10 epochs; float32 Dice; RTX4070 Laptop. Show original CSV log and training/inner curves. Training-only affine/noise, no validation augmentation.','Table B; per-fold provenance/history; docs/AUGMENTATION.md','modified_unet-curves.png + training-log.png','None.','Training loss is soft Dice; reported metrics use binary masks. Curves are inner, not repeated outer selection.'),
      ('Our actual modification','Baseline is paper-inspired Modified U-Net. Insert four additive attention gates before skip concatenation, inspired by Oktay et al. The config changes architecture only.','report Our modification; docs/ATTENTION_UNET.md','attention-architecture.png','None.','Motivation: suppress irrelevant bright tissue. An additional existing architecture alone would not establish our modification.'),
      ('Gate equation and controlled settings','a=sigmoid(psi(ReLU(Ws*s+Wg*g))); gated skip=a*s. Intermediate width half skip channels. 1,967,357 parameters vs 1,945,233 (+22,124). Same split, seeds, augmentation, optimizer, loss, threshold, epochs and checkpoint policy.','kssd/models.py attention_gate; tables/parameters.tex; comparison.json','attention-architecture.png','None.','Matching seed values does not create identical initial weights for different parameter shapes.'),
      ('Baseline versus Attention: negative result','Table E: baseline Dice .7910±.0724 vs Attention .7585±.2205; IoU .6589±.0973 vs .6515±.2864. Delta Dice -.032463. Keep all folds; descriptive comparison only.','Table E and fold table D; report modification','attention-comparison.png','None.','No mean improvement is established, so a superiority claim and improvement ablation are inappropriate.'),
      ('Failure analysis and next experiments','Attention 1143.tif: TP443 FP1344 FN44 Dice.3896; 1068.tif complete miss. Baseline also misses 302.tif. Plausible short budget/tiny foreground/gating dynamics; no proven cause. Focal CV has zero completed folds.','report Discussion and Focal subsection; ATTENTION_RESULTS.md','attention_unet-largest_fp-1143.png','None.','Negative-result analysis is explicitly accepted. Next: longer prespecified budget, multiple seeds, gate placement; no post-hoc outer tuning.'),
      ('Matched small-stone qualitative comparison','126.tif: three GT pixels. Baseline 3/1/0 Dice.8571 IoU.75; Attention 2/0/1 Dice.8 IoU.6667. This case illustrates sensitivity to one pixel; pixel area is not millimetres.','paired-small-stone.png; baseline and Attention figure indexes','paired-small-stone.png','None.','Do not generalize from a hand-selected tiny case or invent clinical size bins.'),
      ('Conclusion, contributions and references','Best canonical mean: Modified U-Net; skip attention lowers mean under ten epochs. Patient IDs/external cohort absent; selected weights local. Cite Bouzon 2025 and Oktay2018 plus architecture papers. State actual member contributions and future work.','report Conclusion, limitations, author contributions; references.bib','Table E or compact conclusion text','Every member actual contribution, speaking part, submission deadline.','Assign every member speaking time and prepare shared answers; do not invent ownership from code history.')]
    brief='# PPT preparation brief: required LaTeX Beamer presentation\n\n'
    brief+='This filename is for teammates; the submitted slide artifact must be Beamer, not PowerPoint. The prepared deck is `presentation/main.tex` and its compiled PDF. Fifteen slides; rehearse about 10 minutes total (40 seconds average), plus five minutes Q&A. Slides share `report/figures/`, avoiding redundant asset copies.\n\n'
    for i,(title,content,source,figure,missing,explanation) in enumerate(slides,1):
        brief+=f'## Slide {i}: {title}\n\n**Exact factual content and numbers:** {content}\n\n**Equation:** '+('Gate equation as above.' if i==11 else 'Dice=2TP/(2TP+FP+FN); IoU=TP/(TP+FP+FN).' if i==2 else 'None required.')+'\n\n'
        brief+=f'**Recommended figure/table:** {figure}; full asset path `report/figures/`.\n\n**Source:** {source}.\n\n**Presenter understanding:** {explanation}\n\n**Team input still required:** {missing}\n\n'
    brief+=all_tables+'\n\n## Examiner questions and evidence-based answers\n\n'
    qa=[('Did you reproduce 97% Dice?','No. That is the paper result from 150 epochs. Our verified baseline mean is .7910 under ten epochs and a different corrected/grouped protocol.'),
        ('Where is the test set?','Each outer fold is untouched during training and checkpoint selection. An inner group split selects the checkpoint. There is no extra external project cohort.'),
        ('Are these patient-independent folds?','No patient IDs exist in the export. Content similarity clusters reduce obvious leakage but do not establish patient independence.'),
        ('What did you modify?','Four same-resolution additive spatial gates on the baseline skip connections, adapted from Oktay et al.; not the invention of Attention U-Net.'),
        ('Did the modification improve?','No mean improvement: Dice delta -.032463. Individual folds vary. We retain failures and make no significance claim.'),
        ('Why no component ablation?','The guideline makes improvement ablation conditional. No improvement is claimed. Baseline/no gates versus all gates is the completed controlled comparison, not a gate-level causal study.'),
        ('Why threshold masks at 127?','Decoded intensities occur in 0-9 and 247-255. >0 adds faint background artifacts. Any integer threshold 9-246 gives the same corrected binary mask.'),
        ('How were metrics/checkpoints reconciled?','All hard scores use probability>=.5; earliest inner hard-Dice maximum selects one checkpoint. Integer TP/FP/FN counts and selected-weight hashes were independently checked.'),
        ('Can you reproduce from GitHub?','Saved-record audits and Attention masks can be checked. New inference requires official data and recovered local H5 weights or retraining; selected weights are not published.'),
        ('Does high recall prove good segmentation?','No: TransUNet 688.tif has recall1 but 10,052 false positives and Dice about .0028.'),
        ('Did Focal Tversky work?','NOT YET MEASURED: the interrupted run has zero complete outer folds. Smoke tests show functionality only.'),
        ('Can you measure clinical stone size?','Not from these exported pixel masks alone: no HU, spacing or scan mapping is available; pixel area is only descriptive.'),
        ('Who did what?','Team must supply actual individual contributions; placeholders are explicit and not inferred.')]
    brief+='\n'.join(f'- **{q}** {a}' for q,a in qa)
    write(ROOT/'docs/PPT_PREPARATION_BRIEF.md',brief)
    record={'dataset_rows':dataset_rows,'splits':split_rows,'paper_table3_percent':PAPER,'summaries':summaries,
            'attention_minus_baseline':dict(zip(KEYS,deltas)),
            'training_budget_extrapolation_hours':sum(summaries[a]['total_training_seconds'] for a in PAPER)*15/3600,
            'figure_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FIGS.glob('*.png')}}
    write(ROOT/'reports/compliance-20261013/deliverable_data.json',json.dumps(record,indent=2))
    print(json.dumps({'figures':len(record['figure_hashes']),'tables':len(list(TABLES.glob('*.tex'))),
                      'slides_in_brief':len(slides),'estimated_150epoch_hours':record['training_budget_extrapolation_hours']}))


if __name__=='__main__':main()
