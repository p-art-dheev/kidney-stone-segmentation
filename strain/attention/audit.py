# Independently verify Attention U-Net's saved 50-epoch results.
import hashlib
import json
import math
from pathlib import Path


def audit(output):
    output = Path(output)
    read = lambda p: json.loads(Path(p).read_text())
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    split = read(output / "splits.json")
    ids = split["splits"]
    check(
        "Entire available split",
        split["counts"] == dict(train=453, validation=288, test=90),
    )
    manifest = read("reports/dataset/manifest.json")
    check(
        "All eligible images retained",
        set(sum(ids.values(), []))
        == {s["id"] for s in manifest["samples"] if s["eligible"]},
    )
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        check(a + "/" + b + " IDs disjoint", not set(ids[a]) & set(ids[b]))
        check(
            a + "/" + b + " content groups disjoint",
            not set(split["groups"][a]) & set(split["groups"][b]),
        )
    folder = output / "attention_unet"
    result = read(folder / "result.json")
    history = read(folder / "history.json")
    epochs = read(folder / "epoch_verification.json")
    check(
        "All 50 epochs complete",
        len(epochs) == len(history["loss"]) == result["epochs_completed"] == 50,
    )
    check(
        "Every epoch uses all training images",
        all(
            r["training_samples"] == len(ids["train"])
            and r["train_batches"] == math.ceil(len(ids["train"]) / result["batch"])
            for r in epochs
        ),
    )
    check(
        "Finite histories",
        all(
            len(v) == 50 and all(math.isfinite(x) for x in v) for v in history.values()
        ),
    )
    for key in [
        "loss",
        "hard_mask_dice",
        "hard_mask_iou",
        "hard_mask_precision",
        "hard_mask_recall",
    ]:
        check(key + " logged", key in history and "val_" + key in history)
    lock = read(output / "selection.lock.json")
    check(
        "Selection sealed before test",
        lock["validation_only"] and not lock["test_data_accessed"],
    )
    check(
        "Checkpoint hash sealed",
        hashlib.sha256(Path(result["checkpoint"]).read_bytes()).hexdigest()
        == result["checkpoint_sha256"]
        == lock["selection"]["checkpoint_sha256"],
    )
    check(
        "Best epoch selected from validation",
        result["best_validation_epoch"]
        == max(range(50), key=lambda i: history["val_hard_mask_dice"][i]) + 1,
    )
    for key, partition in [
        ("validation_common", "validation"),
        ("validation_calibrated", "validation"),
        ("test_common", "test"),
        ("test_calibrated", "test"),
    ]:
        metric = result[key]
        rows = metric["per_image"]
        counts = metric["counts"]
        check(
            key + " identifiers",
            len(rows) == len(ids[partition])
            and {r["id"] for r in rows} == set(ids[partition]),
        )
        check(
            key + " pooled counts",
            all(sum(r[k] for r in rows) == counts[k] for k in counts),
        )
        tp, fp, fn, tn = [counts[k] for k in ["tp", "fp", "fn", "tn"]]
        check(
            key + " pixel count", tp + fp + fn + tn == len(ids[partition]) * 256 * 256
        )
        expected = dict(
            hard_mask_dice=2 * tp / (2 * tp + fp + fn),
            hard_mask_iou=tp / (tp + fp + fn),
            hard_mask_precision=tp / (tp + fp) if tp + fp else 0,
            hard_mask_recall=tp / (tp + fn) if tp + fn else 0,
        )
        check(
            key + " arithmetic",
            all(
                math.isclose(metric["micro"][k], v, abs_tol=1e-10)
                for k, v in expected.items()
            ),
        )
        check(
            key + " Dice/IoU",
            math.isclose(
                metric["micro"]["hard_mask_dice"],
                2
                * metric["micro"]["hard_mask_iou"]
                / (1 + metric["micro"]["hard_mask_iou"]),
                abs_tol=1e-10,
            ),
        )
    check(
        "Common threshold",
        result["validation_common"]["threshold"]
        == result["test_common"]["threshold"]
        == 0.5,
    )
    check(
        "Calibrated threshold unchanged",
        result["validation_calibrated"]["threshold"]
        == result["test_calibrated"]["threshold"]
        == lock["selection"]["validation_calibrated"]["threshold"],
    )
    check(
        "Calibration retains validation recall",
        result["validation_calibrated"]["micro"]["hard_mask_recall"]
        >= 0.85 * result["validation_common"]["micro"]["hard_mask_recall"],
    )
    for p, expected in read(output / "protected_files.json").items():
        check(
            "Historical file preserved: " + p,
            hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected,
        )
    check("Curves saved", (folder / "training_curves.png").exists())
    (output / "independent_verification.json").write_text(
        json.dumps(dict(passed=True, check_count=len(checks), checks=checks), indent=2)
        + "\n"
    )
    print(json.dumps(dict(passed=True, check_count=len(checks))))
