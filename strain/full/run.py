# Run the four-model experiment with a shared deadline.
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def protected():
    files = list(
        Path("strain/checkpoints/experiment256_final_20261013").glob("*.weights.h5")
    )
    files += [Path("kssd/models.py"), Path("strain/train.py"), Path("strain/data.py")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def worker(args):
    import tensorflow as tf
    from strain.data import save
    from strain.full.data import prepare, load_partition, BASE
    from strain.full.train import MODELS, pilot, fit_stage, compile_model
    from strain.targeted.diagnose import (
        diagnose,
        predictions,
        evaluate_probabilities,
        sweep,
    )

    output = Path(args.output)
    end = args.started + args.budget - 70
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU required; no full training started")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    protection = protected()
    save(output / "protected_files.json", protection)
    save(
        output / "config.json",
        dict(
            seed=20261013,
            input_size=256,
            target_epochs=args.epochs,
            shared_budget_seconds=args.budget,
            warm_start=True,
            precision="float32",
            threshold=0.5,
            validation_interval=1,
            models=MODELS,
            candidate_losses=["dice_bce", "fp_tversky_bce"],
            candidate_epochs=args.trial_epochs,
            candidate_learning_rate=3e-4,
            reference_learning_rate=1e-4,
            checkpoint_objective="maximum pooled validation Dice at common threshold 0.5",
            order="prioritize U-Net++ and TransUNet",
            mixed_precision_decision="Prior AMP failures; retain measured stable float32",
        ),
    )
    save(
        output / "environment.json",
        dict(
            tensorflow=tf.__version__,
            keras=tf.keras.__version__,
            gpu_details=[tf.config.experimental.get_device_details(g) for g in gpus],
            nvidia_smi=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.free,driver_version",
                    "--format=csv",
                ],
                text=True,
            ),
        ),
    )
    arrays, splits, cache = prepare(args.data, output)
    checkpoints = {
        n: Path(json.loads((BASE / n / "result.json").read_text())["checkpoint"])
        for n in MODELS
    }
    baseline = {}
    timings = {}
    errors = {}
    for name in MODELS:
        folder = output / name
        folder.mkdir(exist_ok=True)
        try:
            if args.resume_from:
                prior = Path(args.resume_from)
                assert (
                    json.loads((prior / "protected_files.json").read_text())
                    == protection
                )
                baseline[name] = json.loads(
                    (prior / name / "baseline_validation.json").read_text()
                )
                measured = json.loads((prior / name / "pilot.json").read_text())
                timings[name] = dict(
                    batch=measured["selected_batch"],
                    estimated_epoch_seconds=measured["estimated_epoch_seconds"],
                )
                shutil.copy2(
                    prior / name / "baseline_validation.json",
                    folder / "baseline_validation.json",
                )
                shutil.copy2(prior / name / "pilot.json", folder / "pilot.json")
                if (prior / name / "diagnostics").exists():
                    shutil.copytree(
                        prior / name / "diagnostics", folder / "diagnostics"
                    )
                save(
                    folder / "reused_pilot.json",
                    dict(
                        source=str(prior),
                        weights_hash_verified=True,
                        updates_discarded=True,
                    ),
                )
                continue
            if name in ["unetpp", "transunet"]:
                diagnose(name, arrays, splits, args.data, folder / "diagnostics")
                diag = json.loads((folder / "diagnostics/diagnosis.json").read_text())
                baseline[name] = diag["validation"]["common"]
            else:
                model = compile_model(name, checkpoints[name], "dice_bce", 1e-4)
                p = predictions(model, arrays["validation"][0])
                baseline[name] = evaluate_probabilities(
                    arrays["validation"][1], p, splits["validation"], 0.5
                )
                del model
            save(folder / "baseline_validation.json", baseline[name])
            timings[name] = pilot(name, checkpoints[name], arrays, folder, end)
        except Exception as error:
            errors[name] = repr(error)
            save(output / "errors.json", errors)
    viable = [n for n in MODELS if n in timings]
    if not viable:
        raise RuntimeError("Every timing pilot failed")
    pilot_overhead = sum(
        2 * args.trial_epochs * timings[n]["estimated_epoch_seconds"]
        for n in viable
        if n in ["unetpp", "transunet"]
    )
    available = end - time.monotonic() - 180 - pilot_overhead
    epoch_budget = min(
        args.epochs,
        max(
            1,
            int(
                available
                / (1.7 * sum(timings[n]["estimated_epoch_seconds"] for n in viable))
            ),
        ),
    )
    save(
        output / "runtime_plan.json",
        dict(
            epoch_budget=epoch_budget,
            requested_epochs=args.epochs,
            available_training_seconds=available,
            estimated_total_fit_seconds=sum(
                epoch_budget * timings[n]["estimated_epoch_seconds"] for n in viable
            ),
            safety_multiplier=1.7,
            pilot_overhead_estimate=pilot_overhead,
            no_training_images_removed=True,
            viable_models=viable,
        ),
    )
    selections = {}
    for index, name in enumerate(viable):
        remaining = viable[index:]
        weights = {
            n: timings[n]["estimated_epoch_seconds"] * epoch_budget + 40
            for n in remaining
        }
        model_end = min(
            end,
            time.monotonic()
            + (end - time.monotonic() - 90) * weights[name] / sum(weights.values()),
        )
        folder = output / name
        batch = timings[name]["batch"]
        baseline_dice = baseline[name]["micro"]["hard_mask_dice"]
        try:
            stages = []
            offset = 0
            initial = checkpoints[name]
            fallback = initial
            best_epoch = 0
            kind = "dice_bce"
            lr = 1e-4 if name in ["modified_unet", "unet3plus"] else 3e-4
            if name in ["unetpp", "transunet"] and epoch_budget >= 10:
                candidates = []
                for candidate_kind in ["dice_bce", "fp_tversky_bce"]:
                    if time.monotonic() > model_end - 90:
                        break
                    stage = fit_stage(
                        name,
                        arrays,
                        splits,
                        folder / candidate_kind,
                        Path("strain/checkpoints")
                        / output.name
                        / (name + "." + candidate_kind + ".best.weights.h5"),
                        checkpoints[name],
                        candidate_kind,
                        lr,
                        batch,
                        args.trial_epochs,
                        model_end,
                        baseline_dice,
                        short=True,
                    )
                    candidates.append(stage)
                if candidates:
                    winner = max(
                        candidates,
                        key=lambda r: r["validation"]["micro"]["hard_mask_dice"],
                    )
                    save(
                        folder / "loss_comparison.json",
                        dict(
                            candidates=candidates,
                            selected_loss=winner["loss"],
                            selection="Common-threshold pooled validation Dice; current loss wins ties",
                        ),
                    )
                    stages = [winner]
                    kind = winner["loss"]
                    offset = winner["epochs_completed"]
                    initial = Path(winner["latest_checkpoint"])
                    fallback = Path(winner["selected_checkpoint"])
                    baseline_dice = winner["validation"]["micro"]["hard_mask_dice"]
                    best_epoch = winner["best_epoch"]
            remaining_epochs = max(0, epoch_budget - offset)
            standard = (
                Path("strain/checkpoints") / output.name / (name + ".best.weights.h5")
            )
            if remaining_epochs and model_end - time.monotonic() > 90:
                stage = fit_stage(
                    name,
                    arrays,
                    splits,
                    folder / "fit",
                    standard,
                    initial,
                    kind,
                    lr,
                    batch,
                    remaining_epochs,
                    model_end,
                    baseline_dice,
                    offset=offset,
                    fallback=fallback,
                    baseline_epoch=best_epoch,
                )
                stages.append(stage)
                chosen = Path(stage["selected_checkpoint"])
                best_epoch = stage["best_epoch"]
            else:
                chosen = fallback
            if chosen.resolve() != standard.resolve():
                standard.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(chosen, standard)
            model = compile_model(name, standard, kind, lr)
            p = predictions(model, arrays["validation"][0])
            y = arrays["validation"][1]
            common = evaluate_probabilities(y, p, splits["validation"], 0.5)
            thresholds, calibrated = sweep(y, p, splits["validation"])

            # Calibrate on validation while retaining at least 85% of its recall.
            recall_floor = 0.85 * common["micro"]["hard_mask_recall"]
            feasible = [
                r for r in thresholds if r["micro"]["hard_mask_recall"] >= recall_floor
            ]
            calibrated = max(
                feasible,
                key=lambda r: (
                    r["micro"]["hard_mask_dice"],
                    -abs(r["threshold"] - 0.5),
                ),
            )
            rows = {}
            for stage in stages:
                h = json.loads(Path(stage["history_path"]).read_text())
                h.setdefault("learning_rate", [stage["learning_rate"]] * len(h["loss"]))
                for k, v in h.items():
                    rows.setdefault(k, []).extend(v)
            save(folder / "history.json", rows)

            import csv

            with (folder / "history.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", *rows])
                for i in range(len(rows.get("loss", []))):
                    writer.writerow([i + 1, *[rows[k][i] for k in rows]])
            selection = dict(
                architecture=name,
                checkpoint=str(standard),
                checkpoint_sha256=hashlib.sha256(standard.read_bytes()).hexdigest(),
                best_validation_epoch=best_epoch,
                epochs_completed=sum(s["epochs_completed"] for s in stages),
                status=(
                    "time_limited"
                    if any(s["status"] == "time_limited" for s in stages)
                    else "complete"
                ),
                partial_epochs=sum(s["partial_epochs"] for s in stages),
                loss=kind,
                learning_rate=lr,
                batch=batch,
                validation_common=common,
                validation_calibrated=calibrated,
                calibrated_recall_floor=recall_floor,
                threshold_objective="Max validation Dice subject to >=85% common-threshold validation recall",
                training_seconds=sum(s["training_seconds"] for s in stages),
                validation_seconds=sum(s["validation_seconds"] for s in stages),
                stages=stages,
                warm_start=str(checkpoints[name]),
                test_data_accessed=False,
            )
            save(folder / "selection.json", selection)
            selections[name] = selection
            del model
        except Exception as error:
            errors[name] = repr(error)
            save(output / "errors.json", errors)

    save(
        # Freeze all choices before test evaluation.
        output / "selection.lock.json",
        dict(
            selections=selections,
            validation_only=True,
            test_data_accessed=False,
            elapsed_seconds=time.monotonic() - args.started,
        ),
    )
    test = load_partition(splits["test"], cache)
    for name, selection in selections.items():
        if time.monotonic() > end:
            break
        try:
            model = compile_model(
                name,
                selection["checkpoint"],
                selection["loss"],
                selection["learning_rate"],
            )
            start = time.monotonic()
            p = predictions(model, test[0])
            duration = time.monotonic() - start
            common = evaluate_probabilities(test[1], p, splits["test"], 0.5)
            calibrated = evaluate_probabilities(
                test[1],
                p,
                splits["test"],
                selection["validation_calibrated"]["threshold"],
            )
            result = dict(
                **{k: v for k, v in selection.items() if k != "test_data_accessed"},
                test_common=common,
                test_calibrated=calibrated,
                evaluation_seconds=duration,
                test_data_accessed=True,
                inference_scope="One prediction pass; same probabilities for both thresholds",
            )
            save(output / name / "result.json", result)
            del model
        except Exception as error:
            errors[name] = repr(error)
            save(output / "errors.json", errors)
    assert protection == protected(), "Historical code/checkpoint changed"
    save(output / "protection_verification.json", dict(passed=True, hashes=protection))
    save(
        output / "worker_complete.json",
        dict(elapsed_seconds=time.monotonic() - args.started, errors=errors),
    )


def report(output, started, status, budget, deadline_unix=None):
    from strain.data import save
    from strain.plot_history import plot

    plot(output)
    models = []
    for name in ["modified_unet", "unet3plus", "unetpp", "transunet"]:
        p = output / name / "result.json"
        models.append(
            json.loads(p.read_text())
            if p.exists()
            else dict(architecture=name, status="incomplete", test_common=None)
        )
    attempt_elapsed = time.monotonic() - started
    elapsed = time.time() - (deadline_unix - 3600) if deadline_unix else attempt_elapsed
    save(
        output / "comparison.json",
        dict(
            models=models,
            elapsed_seconds=elapsed,
            supervisor_status=status,
            budget_seconds=budget,
            attempt_elapsed_seconds=attempt_elapsed,
            under_60_minutes=elapsed < 3600,
            primary="Held-out test, threshold 0.5, pooled pixel counts",
            secondary="Validation-calibrated threshold, >=85% of common validation recall",
            limitations="Content groups are not patient IDs. Expanded test differs from previous small-subset table. Warm-started models, not fresh initialization.",
        ),
    )
    text = [
        "# Full training-partition 256 × 256 experiment",
        "",
        f"Actual elapsed: {elapsed:.2f} seconds. Status: {status}.",
        "",
        "Primary held-out test results; one validation-selected checkpoint per model; threshold 0.5.",
        "",
        "| Model | New epochs | Dice | IoU | Precision | Recall | Fit seconds | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in models:
        m = (r.get("test_common") or {}).get("micro", {})
        vals = [
            f"{m[k]:.4f}" if k in m else "unavailable"
            for k in [
                "hard_mask_dice",
                "hard_mask_iou",
                "hard_mask_precision",
                "hard_mask_recall",
            ]
        ]
        fit = r.get("training_seconds", 0) + r.get("validation_seconds", 0)
        text.append(
            "| "
            + " | ".join(
                [
                    r["architecture"],
                    str(r.get("epochs_completed", "—")),
                    *vals,
                    f"{fit:.1f}",
                    r["status"],
                ]
            )
            + " |"
        )
    text += [
        "",
        "Validation and optional calibrated test metrics are saved separately in result.json.",
        "Previous test values used a smaller, different test partition; numerical changes are not a matched improvement estimate.",
        "Old files/checkpoints are hash-protected. Test data was loaded only after all model/threshold choices were sealed.",
    ]
    (output / "summary.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=float, default=3600)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trial-epochs", type=int, default=5)
    parser.add_argument("--resume-from")
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--started", type=float)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(Path("strain/results").resolve()):
        raise ValueError("Output must be inside strain/results")
    if not 120 <= args.budget <= 3600:
        raise ValueError("Budget must be 120–3600 seconds")
    args.output = str(output)
    if args.worker:
        worker(args)
        return
    if args.deadline_unix:
        args.budget = min(args.budget, args.deadline_unix - time.time())
    if args.budget < 120:
        raise RuntimeError("Original deadline has insufficient remaining time")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitignore").write_text(
        "**/examples.png\n**/probabilities.npz\nmatplotlib/\nresize_preview.png\n"
    )
    env = dict(
        os.environ,
        PYTHONHASHSEED="20261013",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="2",
        MPLCONFIGDIR=str(output / "matplotlib"),
    )
    cmd = [
        sys.executable,
        "-m",
        "strain.full.run",
        "--worker",
        "--data",
        args.data,
        "--output",
        str(output),
        "--budget",
        str(args.budget),
        "--started",
        str(started),
        "--epochs",
        str(args.epochs),
        "--trial-epochs",
        str(args.trial_epochs),
    ]
    if args.resume_from:
        cmd += ["--resume-from", args.resume_from]
    if args.deadline_unix:
        cmd += ["--deadline-unix", str(args.deadline_unix)]
    with (output / "worker.log").open("w") as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            code = child.wait(timeout=args.budget - 35)
            status = "finished" if code == 0 else f"worker_exit_{code}"
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            status = "hard_deadline_terminated"
    report(output, started, status, args.budget, args.deadline_unix)
    print((output / "summary.md").read_text())


if __name__ == "__main__":
    main()
