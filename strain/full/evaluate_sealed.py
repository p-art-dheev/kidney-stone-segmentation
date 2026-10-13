# Evaluate the four previously selected checkpoints.
import argparse
import hashlib
import json
import time
from pathlib import Path


def evaluate(output, data):
    import tensorflow as tf
    from strain.full.data import load_partition
    from strain.full.train import MODELS, compile_model
    from strain.targeted.diagnose import predictions, evaluate_probabilities

    output = Path(output)
    lock = json.loads((output / "selection.lock.json").read_text())
    if not lock.get("validation_only") or lock.get("test_data_accessed"):
        raise ValueError(
            "Model/threshold selections are not sealed before test evaluation"
        )
    splits = json.loads((output / "splits.json").read_text())["splits"]
    test = load_partition(
        splits["test"], json.loads((output / "preprocessing.json").read_text())["cache"]
    )
    for name in MODELS:
        selection = lock["selections"][name]
        checkpoint = Path(selection["checkpoint"])
        if (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != selection["checkpoint_sha256"]
        ):
            raise ValueError("Sealed checkpoint hash mismatch: " + name)
        model = compile_model(
            name, checkpoint, selection["loss"], selection["learning_rate"]
        )
        started = time.monotonic()
        probabilities = predictions(model, test[0])
        inference = time.monotonic() - started
        common = evaluate_probabilities(test[1], probabilities, splits["test"], 0.5)
        threshold = selection["validation_calibrated"]["threshold"]
        calibrated = evaluate_probabilities(
            test[1], probabilities, splits["test"], threshold
        )
        result = {k: v for k, v in selection.items() if k != "test_data_accessed"}
        result.update(
            test_common=common,
            test_calibrated=calibrated,
            evaluation_seconds=inference,
            test_data_accessed=True,
            inference_scope="One prediction pass after the validation selection lock; same probabilities at both thresholds",
            evaluation_repair="An earlier completed post-seal prediction pass could not serialize because test_data_accessed was supplied twice; this sealed-only repair reruns evaluation without tuning.",
        )
        (output / name / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        del model, probabilities
        tf.keras.backend.clear_session()
    protection = json.loads((output / "protected_files.json").read_text())
    for filename, digest in protection.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise AssertionError("Historical file hash changed: " + filename)
    (output / "protection_verification.json").write_text(
        json.dumps(dict(passed=True, hashes=protection), indent=2) + "\n"
    )
    print("Held-out evaluation saved for all four sealed checkpoints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluate(args.output, None)
