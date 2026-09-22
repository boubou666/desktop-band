"""Train the tiny multi-label classifier and export it as an ONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from desktop_band.classifier import CLASSIFIER_LABELS, FEATURE_NAMES


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    targets = []
    sources = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        values = sample["features"]
        if isinstance(values, dict):
            values = [values.get(name, 0.0) for name in FEATURE_NAMES]
        features.append(values)
        sources.append(str(sample.get("source", "unknown")))
        labels = set(sample.get("labels", ()))
        targets.append([float(label in labels) for label in CLASSIFIER_LABELS])
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or x.shape[0] < 8:
        raise ValueError("dataset invalide ou trop petit")
    return x, y, np.asarray(sources, dtype=str)


def train(x: np.ndarray, y: np.ndarray, epochs: int, rate: float):
    weights = np.zeros((x.shape[1], y.shape[1]), dtype=np.float32)
    bias = np.zeros((y.shape[1],), dtype=np.float32)
    positives = np.maximum(1.0, y.sum(axis=0))
    positive_weight = np.clip((y.shape[0] - positives) / positives, 1.0, 12.0)
    for _ in range(epochs):
        logits = x @ weights + bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        sample_weight = 1.0 + y * (positive_weight - 1.0)
        error = (probabilities - y) * sample_weight
        weights -= rate * (x.T @ error / x.shape[0] + weights * 0.0005)
        bias -= rate * error.mean(axis=0)
    return weights, bias


def predict_onnx(path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return np.asarray(
        session.run([output_name], {input_name: x.astype(np.float32)})[0],
        dtype=np.float32,
    )


def macro_f1(probabilities: np.ndarray, targets: np.ndarray) -> float:
    predicted = probabilities >= 0.5
    scores = []
    for index in range(targets.shape[1]):
        truth = targets[:, index] >= 0.5
        if not truth.any():
            continue
        tp = float(np.logical_and(predicted[:, index], truth).sum())
        fp = float(np.logical_and(predicted[:, index], ~truth).sum())
        fn = float(np.logical_and(~predicted[:, index], truth).sum())
        scores.append(2.0 * tp / max(1.0, 2.0 * tp + fp + fn))
    return float(np.mean(scores)) if scores else 0.0


def choose_validation_source(sources: np.ndarray, targets: np.ndarray) -> str:
    unique = sorted(set(sources.tolist()))
    if len(unique) < 3:
        raise ValueError("au moins trois morceaux sont requis pour la validation")
    for candidate in reversed(unique):
        training = sources != candidate
        if np.all(targets[training].sum(axis=0) > 0):
            return candidate
    raise ValueError(
        "aucun morceau ne peut être réservé sans retirer une classe de l'entraînement"
    )


def export_onnx(weights: np.ndarray, bias: np.ndarray, output: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["features", "weights", "bias"], ["logits"]),
            helper.make_node("Sigmoid", ["logits"], ["probabilities"]),
        ],
        "gopnik_instrument_classifier",
        [helper.make_tensor_value_info(
            "features", TensorProto.FLOAT, [None, len(FEATURE_NAMES)]
        )],
        [helper.make_tensor_value_info(
            "probabilities", TensorProto.FLOAT, [None, len(CLASSIFIER_LABELS)]
        )],
        [
            numpy_helper.from_array(weights.astype(np.float32), "weights"),
            numpy_helper.from_array(bias.astype(np.float32), "bias"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="gopnik-band",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("desktop_band/models/instrument_classifier.onnx"),
    )
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--rate", type=float, default=0.18)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    x, y, sources = load_dataset(args.dataset)
    validation_source = choose_validation_source(sources, y)
    training = sources != validation_source
    validation = ~training
    weights, bias = train(x[training], y[training], args.epochs, args.rate)
    candidate_validation = 1.0 / (
        1.0 + np.exp(-np.clip(x[validation] @ weights + bias, -20, 20))
    )
    candidate_score = macro_f1(candidate_validation, y[validation])
    baseline_score = None
    if args.baseline and args.baseline.is_file():
        baseline_score = macro_f1(
            predict_onnx(args.baseline, x[validation]), y[validation]
        )
    improved = baseline_score is not None and candidate_score >= baseline_score + 0.01
    # Once the held-out comparison is complete, train the distributable
    # candidate on every corrected track. Promotion is decided by the caller.
    weights, bias = train(x, y, args.epochs, args.rate)
    export_onnx(weights, bias, args.output)
    report = {
        "samples": int(x.shape[0]),
        "tracks": len(set(sources.tolist())),
        "validation_track": validation_source,
        "candidate_macro_f1": round(candidate_score, 6),
        "baseline_macro_f1": (
            round(float(baseline_score), 6) if baseline_score is not None else None
        ),
        "improved": improved,
        "output": str(args.output),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
