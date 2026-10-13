# Train Attention U-Net for exactly 50 epochs with held-out evaluation.
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EPOCHS = 50
SEED = 20261014


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    import gc
    import numpy as np
    import tensorflow as tf
    from kssd.models import build_model
    from kssd.reproducibility import set_seeds
    from kssd.tf_metrics import make_metrics
    from strain.data import save
    from strain.full.data import prepare, load_partition
    from strain.full.train import dataset
    from strain.train import loss
    from strain.targeted.diagnose import evaluate_probabilities, sweep
    from strain.plot_history import plot

    output = Path(args.output)
    folder = output / "attention_unet"
    folder.mkdir()
    checkpoints = Path("strain/checkpoints") / output.name
    preserved = list(Path("strain/checkpoints").rglob("*.weights.h5"))
    preserved += [
        Path("kssd/models.py"),
        Path("strain/train.py"),
        Path("strain/data.py"),
    ]
    protected = {str(p): digest(p) for p in preserved}
    save(output / "protected_files.json", protected)
    checkpoints.mkdir(parents=True, exist_ok=False)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU required for this prioritized run")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    save(
        output / "environment.json",
        dict(
            tensorflow=tf.__version__,
            keras=tf.keras.__version__,
            gpu=[tf.config.experimental.get_device_details(g) for g in gpus],
            nvidia_smi=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.free,driver_version",
                    "--format=csv",
                ],
                text=True,
            ),
            project_process_check="No project training process found in native Windows or WSL before launch; GPU idle",
            architecture_source_sha256=digest("kssd/models.py"),
        ),
    )
    arrays, splits, cache = prepare(args.data, output)
    save(
        output / "status.json",
        dict(status="timing_pilot", epochs_completed=0, required_epochs=EPOCHS),
    )
    save(
        output / "config.json",
        dict(
            architecture="attention_unet",
            epochs=EPOCHS,
            seed=SEED,
            data_shuffle_augmentation_seed=20261013,
            input_shape=[256, 256, 1],
            initialization="Fresh random initialization; no historical weights",
            training_partition_fraction_used=1.0,
            loss="Per-image Dice + 0.25 BCE",
            optimizer="Adam",
            initial_learning_rate=1e-3,
            minimum_learning_rate=1e-5,
            lr_patience=8,
            early_stopping=False,
            validation_every_epoch=True,
            threshold=0.5,
            preprocessing="Reuse verified bilinear images, /255 normalization, nearest-neighbor binary masks",
            augmentation="Paired random horizontal and vertical flips, training only",
            split_policy="Existing frozen, disjoint content-group partition; patient IDs unavailable",
        ),
    )

    def model_for(precision):
        tf.keras.backend.clear_session()
        gc.collect()
        set_seeds(SEED)
        tf.keras.mixed_precision.set_global_policy(precision)
        model = build_model("attention_unet", (256, 256, 1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=loss,
            metrics=make_metrics(0.5),
            jit_compile=False,
        )
        return model

    def predict(model, images, batch):
        probability = np.concatenate(
            [
                model(images[i : i + batch], training=False).numpy()
                for i in range(0, len(images), batch)
            ]
        )
        assert probability.shape == images.shape and np.isfinite(probability).all()
        assert 0 <= probability.min() <= probability.max() <= 1
        return probability

    rows = []
    for precision in ["mixed_float16", "float32"]:
        for batch in [8, 16, 32]:
            model = model_for(precision)
            start = time.monotonic()
            try:
                tf.config.experimental.reset_memory_stats("GPU:0")
                x, y = arrays["train"][0][:batch], arrays["train"][1][:batch]
                steps = []
                for _ in range(8):
                    begin = time.monotonic()
                    metrics = model.train_on_batch(x, y, return_dict=True)
                    steps.append(time.monotonic() - begin)
                    if not all(np.isfinite(v).all() for v in metrics.values()):
                        raise FloatingPointError("Non-finite pilot metric")
                assert (
                    int(model.optimizer.iterations.numpy()) > 0
                ), "No optimizer updates accepted"
                assert all(np.isfinite(v.numpy()).all() for v in model.weights)
                with tf.GradientTape() as tape:
                    value = tf.reduce_mean(loss(y, model(x, training=True)))
                gradients = tape.gradient(value, model.trainable_variables)
                assert all(
                    np.isfinite(g.numpy()).all() for g in gradients if g is not None
                )
                begin = time.monotonic()
                predict(model, arrays["validation"][0], batch)
                validation = time.monotonic() - begin
                estimate = float(
                    np.ceil(len(arrays["train"][0]) / batch) * np.mean(steps[-3:])
                    + validation
                )
                rows.append(
                    dict(
                        precision=precision,
                        batch=batch,
                        stable=True,
                        step_seconds=steps,
                        validation_seconds=validation,
                        estimated_epoch_seconds=estimate,
                        memory=tf.config.experimental.get_memory_info("GPU:0"),
                        optimizer=type(model.optimizer).__name__,
                        total_seconds=time.monotonic() - start,
                    )
                )
            except (
                tf.errors.ResourceExhaustedError,
                tf.errors.InvalidArgumentError,
                FloatingPointError,
                AssertionError,
            ) as error:
                rows.append(
                    dict(
                        precision=precision,
                        batch=batch,
                        stable=False,
                        error=repr(error),
                    )
                )
                save(
                    output / "pilot.json",
                    dict(measurements=rows, updates_discarded=True),
                )
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                break
            save(output / "pilot.json", dict(measurements=rows, updates_discarded=True))
            del model
    viable = [r for r in rows if r["stable"]]
    if not viable:
        raise RuntimeError("No finite GPU pilot configuration")
    chosen = min(viable, key=lambda r: r["estimated_epoch_seconds"])
    save(
        output / "pilot.json",
        dict(measurements=rows, selected=chosen, updates_discarded=True),
    )
    batch = chosen["batch"]
    model = model_for(chosen["precision"])
    save(
        output / "architecture.json",
        dict(
            name=model.name,
            parameters=model.count_params(),
            input_shape=model.input_shape,
            output_shape=model.output_shape,
            output_dtype=model.output.dtype,
            output_activation=model.layers[-1].activation.__name__,
            attention_gates=[
                layer.name
                for layer in model.layers
                if layer.name.endswith("_gated_skip")
            ],
            precision_policy=tf.keras.mixed_precision.global_policy().name,
            optimizer=type(model.optimizer).__name__,
        ),
    )
    assert model.output_shape == (None, 256, 256, 1)
    assert (
        len([layer for layer in model.layers if layer.name.endswith("_gated_skip")])
        == 4
    )
    best = checkpoints / "attention_unet.best.weights.h5"
    latest = checkpoints / "attention_unet.latest.keras"

    class Progress(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.rows = []

        def on_epoch_begin(self, epoch, logs=None):
            self.begin = time.monotonic()
            self.batches = 0
            self.samples = 0

        def on_train_batch_end(self, batch_index, logs=None):
            self.batches += 1
            self.samples += min(batch, len(arrays["train"][0]) - batch_index * batch)
            if not all(np.isfinite(v).all() for v in (logs or {}).values()):
                raise FloatingPointError("Non-finite training metric")

        def on_epoch_end(self, epoch, logs=None):
            assert self.batches == self.params["steps"]
            assert self.samples == len(splits["train"])
            assert all(np.isfinite(v).all() for v in logs.values())
            self.rows.append(
                dict(
                    epoch=epoch + 1,
                    seconds=time.monotonic() - self.begin,
                    training_samples=self.samples,
                    train_batches=self.batches,
                )
            )
            if (epoch + 1) % 10 == 0:
                self.model.save(latest)
            save(folder / "epoch_verification.json", self.rows)
            save(
                output / "status.json",
                dict(
                    status="training",
                    epochs_completed=epoch + 1,
                    required_epochs=EPOCHS,
                    last_epoch_seconds=self.rows[-1]["seconds"],
                    estimated_remaining_training_seconds=(EPOCHS - epoch - 1)
                    * float(np.mean([r["seconds"] for r in self.rows[-5:]])),
                    last_metrics={k: float(v) for k, v in logs.items()},
                    elapsed_seconds=time.time() - args.started,
                ),
            )

    progress = Progress()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best),
            monitor="val_hard_mask_dice",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_hard_mask_dice",
            mode="max",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
        ),
        tf.keras.callbacks.CSVLogger(str(folder / "history.csv")),
        progress,
    ]
    start = time.monotonic()
    history = model.fit(
        dataset(arrays["train"], batch, True),
        validation_data=dataset(arrays["validation"], batch),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    fit_seconds = time.monotonic() - start
    save(folder / "history.json", history.history)
    assert len(history.history["loss"]) == EPOCHS and len(progress.rows) == EPOCHS
    model.load_weights(best)
    probability = predict(model, arrays["validation"][0], batch)
    common = evaluate_probabilities(
        arrays["validation"][1], probability, splits["validation"], 0.5
    )
    thresholds, _ = sweep(arrays["validation"][1], probability, splits["validation"])
    feasible = [
        r
        for r in thresholds
        if r["micro"]["hard_mask_recall"] >= 0.85 * common["micro"]["hard_mask_recall"]
    ]
    calibrated = max(
        feasible,
        key=lambda r: (r["micro"]["hard_mask_dice"], -abs(r["threshold"] - 0.5)),
    )
    selected = dict(
        architecture="attention_unet",
        epochs_completed=EPOCHS,
        best_validation_epoch=int(np.argmax(history.history["val_hard_mask_dice"])) + 1,
        checkpoint=str(best),
        checkpoint_sha256=digest(best),
        latest_full_model=str(latest),
        batch=batch,
        precision=chosen["precision"],
        fit_seconds=fit_seconds,
        validation_common=common,
        validation_calibrated=calibrated,
        test_data_accessed=False,
    )
    save(folder / "selection.json", selected)
    save(
        output / "selection.lock.json",
        dict(selection=selected, validation_only=True, test_data_accessed=False),
    )
    # Materialize test data only after checkpoint and threshold choices are sealed.
    test = load_partition(splits["test"], cache)
    start = time.monotonic()
    probability = predict(model, test[0], batch)
    result = dict(selected)
    result.update(
        test_data_accessed=True,
        evaluation_seconds=time.monotonic() - start,
        test_common=evaluate_probabilities(test[1], probability, splits["test"], 0.5),
        test_calibrated=evaluate_probabilities(
            test[1], probability, splits["test"], calibrated["threshold"]
        ),
    )
    save(folder / "result.json", result)
    plot(output)
    for path, expected in protected.items():
        assert digest(path) == expected, path
    from strain.attention.audit import audit

    audit(output)
    elapsed = time.time() - args.started
    save(
        output / "status.json",
        dict(
            status="complete",
            epochs_completed=EPOCHS,
            required_epochs=EPOCHS,
            elapsed_seconds=elapsed,
            fit_seconds=fit_seconds,
            precision=chosen["precision"],
            batch=batch,
        ),
    )
    micro = result["test_common"]["micro"]
    text = [
        "# Attention U-Net — completed 50-epoch run",
        "",
        f"Actual wall-clock time: {elapsed/60:.2f} minutes. Fit and epoch validation: {fit_seconds:.2f} seconds.",
        "",
        f"All 50 complete epochs used every training image. Selected validation epoch: {selected['best_validation_epoch']}. Fresh random initialization; precision: {chosen['precision']}; batch: {batch}.",
        "",
        "Primary held-out test metrics at threshold 0.5, pooled pixel counts:",
        "",
        "| Dice | IoU | Precision | Recall |",
        "|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            f"{micro[k]:.4f}"
            for k in [
                "hard_mask_dice",
                "hard_mask_iou",
                "hard_mask_precision",
                "hard_mask_recall",
            ]
        )
        + " |",
        "",
        "Validation and optional validation-calibrated test results are saved separately in attention_unet/result.json. Both test thresholds share one prediction pass. All choices were sealed before test loading.",
        "",
        "Frozen split: 453 training / 288 validation / 90 test images, all 831 eligible unique samples. Seven existing duplicate exclusions are retained; 100% of the training partition is fitted. Content-group separation does not prove patient independence without patient IDs.",
        "",
        "Preprocessing reuses verified 256-pixel bilinear CT images, uint8 /255 normalization and nearest-neighbor binary masks. No entire mask vanished, but small components disappeared in 50 original images; IDs remain in preprocessing.json.",
        "",
        f"Best weights: `{best}`. Latest full model with optimizer: `{latest}`. Training/validation loss and all four metrics are saved to CSV/JSON with curves. Pilot updates were discarded.",
        "",
        "Historical files and checkpoints passed their hash checks. Independent verification is recorded in independent_verification.json. No competing project experiment was running before launch; no unrelated process was stopped.",
        "",
        "Rerun from repository root:",
        "",
        "```bash",
        f"python -m strain.attention.run --data {args.data} --output strain/results/attention256_new_run",
        "```",
    ]
    (output / "report.md").write_text("\n".join(text) + "\n")
    print((output / "report.md").read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--started", type=float)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(Path("strain/results").resolve()):
        raise ValueError("Output must be inside strain/results")
    if args.worker:
        try:
            worker(args)
        except BaseException as error:
            (output / "failure.json").write_text(
                json.dumps(
                    dict(error=repr(error), elapsed_seconds=time.time() - args.started),
                    indent=2,
                )
            )
            raise
        return
    started = time.time()
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitignore").write_text("matplotlib/\nresize_preview.png\n")
    env = dict(
        os.environ,
        PYTHONHASHSEED=str(SEED),
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="2",
        MPLCONFIGDIR=str(output / "matplotlib"),
    )
    command = [
        sys.executable,
        "-m",
        "strain.attention.run",
        "--worker",
        "--data",
        args.data,
        "--output",
        str(output),
        "--started",
        str(started),
    ]
    with (output / "worker.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        code = child.wait()
    if code:
        raise SystemExit(code)
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
