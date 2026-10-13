# Plot per-epoch training and validation loss and metrics.
import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(output):
    for path in Path(output).glob("*/history.json"):
        history = json.loads(path.read_text())
        epochs = range(1, len(history["loss"]) + 1)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        keys = [
            "loss",
            "hard_mask_dice",
            "hard_mask_iou",
            "hard_mask_precision",
            "hard_mask_recall",
        ]
        labels = ["Loss", "Dice", "IoU", "Precision", "Recall"]
        best = 1 + max(
            range(len(history["val_hard_mask_dice"])),
            key=lambda i: history["val_hard_mask_dice"][i],
        )
        selection = path.parent / "selection.json"
        if selection.exists():
            best = json.loads(selection.read_text())["best_validation_epoch"]
        for ax, key, label in zip(axes.flat, keys, labels):
            ax.plot(epochs, history[key], label="Training")
            ax.plot(epochs, history["val_" + key], label="Validation")
            ax.axvline(best, color="#777777", linestyle=":", label="Selected epoch")
            ax.set(xlabel="Epoch", ylabel=label)
            if key != "loss":
                ax.set_ylim(0, 1.02)
            ax.legend()
            ax.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        names = dict(
            modified_unet="Modified U-Net",
            unet3plus="U-Net3+",
            unetpp="U-Net++",
            transunet="TransUNet",
            attention_unet="Attention U-Net",
        )
        fig.suptitle(
            names.get(path.parent.name, path.parent.name) + " — Training history"
        )
        fig.tight_layout()
        fig.savefig(path.parent / "training_curves.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    plot(parser.parse_args().output)
