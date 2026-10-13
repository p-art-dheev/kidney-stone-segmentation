# Build the final comparison and experiment report.
import argparse
import json
import time
from pathlib import Path
from strain.full.audit_results import audit


def finalize(output):
    output = Path(output)
    read = lambda p: json.loads(Path(p).read_text())
    speedups = {
        n: read(Path("strain/results/experiment256_final_20261013") / n / "pilot.json")[
            "measured_speedup"
        ]
        for n in ["modified_unet", "unet3plus", "unetpp", "transunet"]
    }
    comparison = read(output / "comparison.json")
    models = comparison["models"]
    clock = output / "experiment_clock.json"
    if clock.exists():
        comparison.setdefault(
            "first_finalization_attempt_seconds", comparison["elapsed_seconds"]
        )
        comparison["training_worker_seconds"] = read(output / "worker_complete.json")[
            "elapsed_seconds"
        ]
        comparison["elapsed_seconds"] = (
            time.time() - read(clock)["original_started_unix"]
        )
        comparison["under_60_minutes"] = comparison["elapsed_seconds"] < 3600
        comparison["elapsed_scope"] = (
            "Original start through independent audit and final report preparation"
        )
        comparison["models"] = [
            read(output / name / "result.json")
            for name in ["modified_unet", "unet3plus", "unetpp", "transunet"]
        ]
        if any(r.get("test_common") is None for r in comparison["models"]):
            raise AssertionError("One or more held-out test results are missing")
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        (output / "errors.json").write_text(
            json.dumps(
                dict(
                    unresolved={},
                    initial_finalization_errors=read(output / "worker_complete.json")[
                        "errors"
                    ],
                    status="All four sealed-checkpoint test evaluations were repaired and completed",
                ),
                indent=2,
            )
            + "\n"
        )
    models = comparison["models"]
    audit(output)
    names = dict(
        modified_unet="Modified U-Net",
        unet3plus="U-Net3+",
        unetpp="U-Net++",
        transunet="TransUNet",
    )
    text = [
        "# Completed four-model experiment",
        "",
        f"Measured overall elapsed through final audit/report: **{comparison['elapsed_seconds']/60:.1f} minutes**. Under 60 minutes: **{comparison['under_60_minutes']}**. The original supervisor reached its first report attempt at {comparison['first_finalization_attempt_seconds']/60:.1f} minutes. That attempt trained all models but lost the test-result JSON to a duplicate-field serialization error; the sealed-only evaluation repair and final audit completed after the session was resumed.",
        "",
        "Each model uses the full frozen training partition: **453 training, 288 validation, 90 test images**, all 831 eligible unique images; training fraction used is 100%. Seven original pairs excluded by the existing duplicate/eligibility policy were not newly removed. Content groups are disjoint; patient independence cannot be established without patient IDs.",
        "",
        "The user authorized reducing the original 50-epoch target to **20 additional epochs per selected trajectory**. U-Net++ and TransUNet also each have a separately logged three-epoch losing loss trial. These are warm starts from each architecture’s historical checkpoint, not training from scratch.",
        "",
        "## Held-out test, common threshold 0.5",
        "",
        "| Model | New selected epochs | Dice | IoU | Precision | Recall | Selected fit + validation (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in models:
        m = r["test_common"]["micro"]
        text.append(
            "| "
            + " | ".join(
                [
                    names[r["architecture"]],
                    str(r["epochs_completed"]),
                    *[
                        f"{m[k]:.4f}"
                        for k in [
                            "hard_mask_dice",
                            "hard_mask_iou",
                            "hard_mask_precision",
                            "hard_mask_recall",
                        ]
                    ],
                    f"{r['training_seconds']+r['validation_seconds']:.1f}",
                ]
            )
            + " |"
        )
    text += [
        "",
        "These are pooled pixel metrics on the same held-out test masks. The earlier 16-image test differs from this 90-image test, so changes from the old table are not a matched estimate of improvement. Final reported metrics use one prediction pass per sealed checkpoint; both thresholds use the same probabilities. An earlier post-seal test pass failed while saving JSON because of a duplicate field, so final evaluation repeated inference without changing choices or tuning on test results.",
        "",
        "## Validation used for selection",
        "",
        "| Model | Selected epoch | Dice | IoU | Precision | Recall | Loss |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in models:
        m = r["validation_common"]["micro"]
        text.append(
            "| "
            + " | ".join(
                [
                    names[r["architecture"]],
                    str(r["best_validation_epoch"]),
                    *[
                        f"{m[k]:.4f}"
                        for k in [
                            "hard_mask_dice",
                            "hard_mask_iou",
                            "hard_mask_precision",
                            "hard_mask_recall",
                        ]
                    ],
                    r["loss"],
                ]
            )
            + " |"
        )
    text += [
        "",
        "## Optional validation-calibrated test results",
        "",
        "Secondary results only. Threshold maximizes validation Dice while preserving at least 85% of the selected checkpoint’s validation recall at 0.5. No test threshold tuning.",
        "",
        "| Model | Threshold | Test Dice | Test IoU | Test precision | Test recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in models:
        m = r["test_calibrated"]["micro"]
        text.append(
            "| "
            + " | ".join(
                [
                    names[r["architecture"]],
                    str(r["test_calibrated"]["threshold"]),
                    *[
                        f"{m[k]:.4f}"
                        for k in [
                            "hard_mask_dice",
                            "hard_mask_iou",
                            "hard_mask_precision",
                            "hard_mask_recall",
                        ]
                    ],
                ]
            )
            + " |"
        )
    text += [
        "",
        "## Diagnoses and limitations",
        "",
        "All 838 original pairs and cached resized pairs were rechecked. Images are display-encoded uint8 CT, normalized by /255 after bilinear resizing, not Hounsfield-unit windowing. Masks decode to grayscale 0–9 background and 247–255 foreground, threshold >127, then nearest-neighbor binary resize. Images/masks receive paired training-only flips; validation/test preprocessing is deterministic. No originals or historical checkpoints were modified.",
        "",
        "No complete foreground mask disappeared at 256 pixels, but 50 images lost at least one small connected component. IDs are recorded in preprocessing.json. Resizing can reduce the representability of tiny stones; metrics are evaluated honestly against the nearest-neighbor masks, not against altered or filtered ground truth.",
        "",
        "U-Net++ and TransUNet have valid 256×256 single sigmoid heads, compatible probability losses, and independently verified metrics. U-Net++ decoder connections and TransUNet transformer/token/decoder structure are recorded in diagnostics; no architecture substitution or new deep supervision was introduced. Extreme foreground sparsity and false positives are confirmed. On baseline validation, ground truth foreground occupied 0.068977% of pixels; U-Net++ predicted 0.099150% and TransUNet 0.146235%. False negatives also remain. Threshold changes alone did not solve the low Dice. Underfitting/schedule and generalization problems are plausible, but no single exclusive cause is proven.",
        "",
        "Current Dice+BCE and FP-weighted Tversky+BCE trials used identical weights, split, LR and augmentation. Loss choice used validation only. Training loss improvement does not imply improved generalization. The selected loss and every candidate outcome are recorded in loss_comparison.json; old weights remained eligible at epoch zero.",
        "",
        "Worker PID 338 received SIGTERM while awaiting the prior experiment plan; supervisor PID 301 exited. The stop was verified: no remaining project training process, GPU utilization 0%, memory 9 MiB. The first full attempt failed before fitting because imported-source edits confused AutoGraph tracing. It is preserved under full256_20261013. The corrected restart disables unnecessary conversion for paired TensorFlow augmentation and retains the original absolute deadline.",
        "",
        "Matched historical 256-vs-512 compute pilots measured speedups of "
        + ", ".join(f"{names[k]} {v:.2f}×" for k, v in speedups.items())
        + ". They used the same cached samples, with the 256-pixel samples upsampled for 512 timing; this is a compute comparison, not an accuracy comparison. Current full-partition pilots selected batches 16 for Modified U-Net/U-Net++/TransUNet and 8 for U-Net3+; batch 16 exceeded U-Net3+ memory. Float32 is retained after prior mixed-precision instability.",
        "",
        "## Timing, logs and checkpoints",
        "",
        "Per-model history.csv/history.json contain training and validation loss, Dice, IoU, precision and recall for every selected-trajectory epoch. Losing trials remain in separate folders. training_curves.png has architecture-only titles and no dataset-size labels. Best validation checkpoints are local under strain/checkpoints/"
        + output.name
        + "/; they are ignored by Git. Raw CT panels and probabilities also remain local.",
        "",
        "| Model | Checkpoint | Selected-stage elapsed (s) | Other loss-trial elapsed (s) | Test inference (s) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in models:
        trial = output / r["architecture"] / "loss_comparison.json"
        other = 0
        if trial.exists():
            other = sum(
                s["total_seconds"]
                for s in read(trial)["candidates"]
                if s["loss"] != r["loss"]
            )
        text.append(
            f"| {names[r['architecture']]} | `{r['checkpoint']}` | {sum(s['total_seconds'] for s in r['stages']):.1f} | {other:.1f} | {r['evaluation_seconds']:.1f} |"
        )
    text += [
        "",
        "Overall elapsed includes preprocessing, pilots, initialization, failed attempt, debugging/restart gap, threshold evaluation, fitting, testing and supervisor curve/report generation. Stage timing includes initialization and checkpoint validation; fit timing in the first table counts only selected fit plus epoch validation. Full JSON contains actual partial epochs, batches and checkpoint hashes.",
        "",
        "Independent verification is in independent_verification.json. Historical file hashes, split coverage, every per-image pooled metric, Dice/IoU identity, sealed checkpoint hashes, finite histories and all curves were checked.",
        "",
        "Changed scripts: strain/full/data.py, train.py, run.py, test_full.py, audit_results.py, finalize_results.py and README.md; strain/plot_history.py. Existing model implementations were reused. Source and compact reports are pushed to the approved repository; ignored CT panels/caches/weights remain local.",
        "",
        "Fresh rerun command from repository root (starts a new budget):",
        "",
        "```bash",
        "/var/tmp/kssd-gpu312-20261012/bin/python -m strain.full.run --data /var/tmp/kssd-workspace-20261012/data/kssd2025/data --output strain/results/full256_new_run --epochs 20 --trial-epochs 3",
        "```",
        "",
        "The exact restart command retaining this experiment’s deadline is in strain/full/README.md. No 0.90 Dice or 50 completed epochs is claimed.",
    ]
    report = "\n".join(text) + "\n"
    (output / "report.md").write_text(report)
    (output / "summary.md").write_text(report)
    (output / "resizing_compute_speedups.json").write_text(
        json.dumps(
            dict(
                speedups=speedups,
                evidence="Existing matched 256/512 compute pilots, same cached training samples upsampled for 512; not an accuracy comparison",
            ),
            indent=2,
        )
        + "\n"
    )
    print(str(output / "report.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    finalize(parser.parse_args().output)
