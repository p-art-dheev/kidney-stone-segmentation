import numpy as np
import pytest
from kssd.figures import comparison_plot,prediction_figure,select_cases,training_plot,verify_prediction
from kssd.metrics import binary_metrics


def synthetic_pair():
    image=np.tile(np.linspace(0,1,32,dtype=np.float32),(32,1))[...,None]
    truth=np.zeros_like(image);truth[14:18,14:18]=1
    probability=np.zeros_like(image);probability[15:19,15:19]=.9
    row={'id':'synthetic-only.tif','stone_area':16,**binary_metrics(truth,probability)}
    return image,truth,probability,row


def test_restored_predictions_must_match_saved_counts():
    image,truth,probability,row=synthetic_pair()
    assert verify_prediction(truth,probability,row,.5)['tp']==9
    with pytest.raises(ValueError,match='counts differ'):verify_prediction(truth,np.zeros_like(probability),row,.5)


def test_select_cases_deterministic_and_keeps_failures():
    _,_,_,a=synthetic_pair();b={**a,'id':'b.tif','hard_mask_dice':0,'fn':16,'fp':100,'stone_area':3}
    selected=select_cases([a,b])
    assert selected['highest_dice']['id']==a['id']
    assert selected['lowest_dice']['id']==b['id']
    assert selected['largest_fp']['id']==b['id'] and selected['smallest_area']['id']==b['id']
    assert select_cases([b,a])==selected


def test_render_verified_synthetic_case(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from PIL import Image
    image,truth,probability,row=synthetic_pair();output=tmp_path/'synthetic-only.png'
    prediction_figure(image,truth,probability,row,.5,output,'SYNTHETIC FUNCTIONAL TEST — no research result')
    with Image.open(output) as rendered:
        assert rendered.width>=2000 and rendered.height>=500
    assert output.stat().st_size>10000


def test_comparison_and_history_renderers(tmp_path):
    import json
    import matplotlib
    matplotlib.use('Agg')
    from kssd.reporting import METRICS
    summaries=[{'architecture':f'SYNTHETIC_{i}',
                'fold_mean':{m:.25+i*.1 for m in METRICS},
                'fold_sample_std':{m:.02 for m in METRICS}} for i in range(4)]
    comparison_plot(summaries,tmp_path/'synthetic-comparison.png')
    root=tmp_path/'SYNTHETIC-HISTORY';root.mkdir()
    for fold in range(1,6):
        folder=root/f'fold-{fold}';folder.mkdir()
        (folder/'history.json').write_text(json.dumps({'loss':[.8,.6],'val_loss':[.9,.7],
                                                     'hard_mask_dice':[.1,.4],'val_hard_mask_dice':[.2,.3]}))
    training_plot(root,tmp_path/'synthetic-curves.png')
    assert (tmp_path/'synthetic-comparison.png').stat().st_size>10000
    assert (tmp_path/'synthetic-curves.png').stat().st_size>10000
