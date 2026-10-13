"""Plot persisted epoch losses and hard-mask metrics for every completed model."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(output):
    for path in Path(output).glob('*/history.json'):
        history = json.loads(path.read_text())
        epochs = range(1, len(history['loss'])+1)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, key in zip(axes, ['loss','hard_mask_dice','hard_mask_iou']):
            ax.plot(epochs, history[key], label='Training')
            ax.plot(epochs, history['val_'+key], label='Validation')
            ax.set(xlabel='Epoch', ylabel=key)
            ax.legend()
            ax.grid(alpha=.2)
        fig.suptitle(path.parent.name+' | 256 × 256 | test excluded')
        fig.tight_layout()
        fig.savefig(path.parent/'training_curves.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    plot(parser.parse_args().output)
