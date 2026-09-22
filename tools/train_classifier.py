"""Train the tiny multi-label classifier and export it as an ONNX graph."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

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


def _write_progress(path: Path | None, **values) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def train(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    rate: float,
    progress=None,
):
    weights = np.zeros((x.shape[1], y.shape[1]), dtype=np.float32)
    bias = np.zeros((y.shape[1],), dtype=np.float32)
    positives = np.maximum(1.0, y.sum(axis=0))
    positive_weight = np.clip((y.shape[0] - positives) / positives, 1.0, 12.0)
    for epoch in range(epochs):
        logits = x @ weights + bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        sample_weight = 1.0 + y * (positive_weight - 1.0)
        error = (probabilities - y) * sample_weight
        weights -= rate * (x.T @ error / x.shape[0] + weights * 0.0005)
        bias -= rate * error.mean(axis=0)
        if progress is not None and (epoch % 50 == 0 or epoch + 1 == epochs):
            progress((epoch + 1) / epochs)
    return weights, bias


def augment_dataset(
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    copies: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create conservative feature-space gain/EQ/noise variants."""

    if copies <= 0:
        return x, y, sources
    random = np.random.default_rng(seed)
    variants = [x]
    targets = [y]
    names = [sources]
    for index in range(copies):
        gain = random.uniform(0.82, 1.18, size=(x.shape[0], 1)).astype(np.float32)
        equalizer = random.uniform(0.91, 1.09, size=x.shape).astype(np.float32)
        noise = random.normal(0.0, 0.012, size=x.shape).astype(np.float32)
        transformed = np.clip(x * gain * equalizer + noise, 0.0, 1.0)
        variants.append(transformed)
        targets.append(y)
        names.append(np.char.add(sources, f"#aug{index + 1}"))
    return np.concatenate(variants), np.concatenate(targets), np.concatenate(names)


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
    scores = per_role_f1(probabilities, targets)
    return float(np.mean(list(scores.values()))) if scores else 0.0


def per_role_f1(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    if thresholds is None:
        thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    predicted = probabilities >= thresholds.reshape(1, -1)
    scores = {}
    for index, label in enumerate(CLASSIFIER_LABELS):
        truth = targets[:, index] >= 0.5
        if not truth.any() or truth.all():
            continue
        tp = float(np.logical_and(predicted[:, index], truth).sum())
        fp = float(np.logical_and(predicted[:, index], ~truth).sum())
        fn = float(np.logical_and(~predicted[:, index], truth).sum())
        scores[label] = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
    return scores


def confusion_by_role(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, dict[str, int]]:
    predicted = probabilities >= thresholds.reshape(1, -1)
    result = {}
    for index, label in enumerate(CLASSIFIER_LABELS):
        truth = targets[:, index] >= 0.5
        result[label] = {
            "tp": int(np.logical_and(predicted[:, index], truth).sum()),
            "fp": int(np.logical_and(predicted[:, index], ~truth).sum()),
            "fn": int(np.logical_and(~predicted[:, index], truth).sum()),
            "tn": int(np.logical_and(~predicted[:, index], ~truth).sum()),
        }
    return result


def optimal_thresholds(
    probabilities: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    candidates = np.arange(0.15, 0.86, 0.05, dtype=np.float32)
    for index in range(targets.shape[1]):
        truth = targets[:, index] >= 0.5
        if not truth.any():
            continue
        best_score = -1.0
        for candidate in candidates:
            predicted = probabilities[:, index] >= candidate
            tp = float(np.logical_and(predicted, truth).sum())
            fp = float(np.logical_and(predicted, ~truth).sum())
            fn = float(np.logical_and(~predicted, truth).sum())
            score = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
            if score > best_score:
                best_score = score
                thresholds[index] = candidate
    return thresholds


def macro_f1_at(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    scores = per_role_f1(probabilities, targets, thresholds)
    return float(np.mean(list(scores.values()))) if scores else 0.0


def _legacy_macro_f1(probabilities: np.ndarray, targets: np.ndarray) -> float:
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


def choose_validation_sources(
    sources: np.ndarray,
    targets: np.ndarray,
    fraction: float = 0.15,
) -> list[str]:
    """Reserve whole source groups while covering every available role."""

    unique = sorted(set(sources.tolist()))
    if len(unique) < 3:
        raise ValueError("au moins trois morceaux sont requis pour la validation")
    group_rows = {name: sources == name for name in unique}
    group_labels = {
        name: set(np.flatnonzero(targets[mask].sum(axis=0) > 0).tolist())
        for name, mask in group_rows.items()
    }
    required = set(np.flatnonzero(targets.sum(axis=0) > 0).tolist())
    selected: list[str] = []

    def can_reserve(candidate: str) -> bool:
        reserved = np.isin(sources, [*selected, candidate])
        return bool(np.all(targets[~reserved].sum(axis=0) > 0))

    uncovered = set(required)
    while uncovered:
        candidates = [
            name for name in unique
            if name not in selected and can_reserve(name)
            and group_labels[name] & uncovered
        ]
        if not candidates:
            missing = ", ".join(CLASSIFIER_LABELS[index] for index in uncovered)
            raise ValueError(
                "impossible de réserver une validation couvrant : " + missing
            )
        candidate = max(candidates, key=lambda name: (
            len(group_labels[name] & uncovered),
            -int(group_rows[name].sum()),
            name,
        ))
        selected.append(candidate)
        uncovered -= group_labels[candidate]

    target_count = max(3, math.ceil(len(unique) * max(0.05, min(0.3, fraction))))
    target_count = min(target_count, len(unique) - 2)
    while len(selected) < target_count:
        candidates = [
            name for name in unique
            if name not in selected and can_reserve(name)
        ]
        if not candidates:
            break
        # Extra groups favor smaller holdouts and deterministic diversity.
        candidate = min(candidates, key=lambda name: (
            int(group_rows[name].sum()), name
        ))
        selected.append(candidate)
    validation = np.isin(sources, selected)
    if not np.all(targets[validation].sum(axis=0) > 0):
        raise ValueError("la validation ne couvre pas tous les instruments")
    return selected


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
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1917)
    args = parser.parse_args()
    x, y, sources = load_dataset(args.dataset)
    validation_sources = choose_validation_sources(sources, y)
    validation = np.isin(sources, validation_sources)
    training = ~validation
    train_x, train_y, train_sources = augment_dataset(
        x[training], y[training], sources[training],
        max(0, args.augment_copies), args.seed,
    )
    started = time.monotonic()
    weights, bias = train(
        train_x, train_y, args.epochs, args.rate,
        lambda value: _write_progress(
            args.progress_file,
            phase="validate",
            progress=value * 0.55,
            elapsed_seconds=round(time.monotonic() - started, 1),
            eta_seconds=round(
                (time.monotonic() - started) * (1.0 - value) / max(value, 1e-4), 1
            ),
            current_track=f"{len(validation_sources)} morceaux de validation",
        ),
    )
    candidate_validation = 1.0 / (
        1.0 + np.exp(-np.clip(x[validation] @ weights + bias, -20, 20))
    )
    thresholds = optimal_thresholds(candidate_validation, y[validation])
    candidate_score = macro_f1(candidate_validation, y[validation])
    calibrated_score = macro_f1_at(
        candidate_validation, y[validation], thresholds
    )
    candidate_roles = per_role_f1(candidate_validation, y[validation])
    calibrated_roles = per_role_f1(
        candidate_validation, y[validation], thresholds
    )
    baseline_score = None
    baseline_roles = {}
    if args.baseline and args.baseline.is_file():
        baseline_validation = predict_onnx(args.baseline, x[validation])
        baseline_score = macro_f1(baseline_validation, y[validation])
        baseline_roles = per_role_f1(baseline_validation, y[validation])
    regressions = {
        role: candidate_roles[role] - baseline_roles[role]
        for role in candidate_roles.keys() & baseline_roles.keys()
    }
    severe = [role for role, delta in regressions.items() if delta < -0.08]
    reasons = []
    if baseline_score is None:
        reasons.append("aucun modèle de référence")
    elif candidate_score < baseline_score + 0.01:
        reasons.append("gain macro-F1 inférieur à 1 point")
    if severe:
        reasons.append("régression par rôle : " + ", ".join(severe))
    improved = not reasons
    # Once the held-out comparison is complete, train the distributable
    # candidate on every corrected track. Promotion is decided by the caller.
    all_x, all_y, _all_sources = augment_dataset(
        x, y, sources, max(0, args.augment_copies), args.seed + 1
    )
    weights, bias = train(
        all_x, all_y, args.epochs, args.rate,
        lambda value: _write_progress(
            args.progress_file,
            phase="train",
            progress=0.55 + value * 0.45,
            elapsed_seconds=round(time.monotonic() - started, 1),
            eta_seconds=round(
                (time.monotonic() - started) * (1.0 - value)
                / max(0.55 + value * 0.45, 1e-4), 1
            ),
            current_track="tous les morceaux",
        ),
    )
    export_onnx(weights, bias, args.output)
    report = {
        "samples": int(x.shape[0]),
        "tracks": len(set(sources.tolist())),
        "validation_track": ", ".join(validation_sources),
        "validation_tracks": validation_sources,
        "candidate_macro_f1": round(candidate_score, 6),
        "candidate_calibrated_macro_f1": round(calibrated_score, 6),
        "baseline_macro_f1": (
            round(float(baseline_score), 6) if baseline_score is not None else None
        ),
        "improved": improved,
        "promotion_reasons": reasons or ["gain validé sans régression majeure"],
        "thresholds": {
            label: round(float(value), 3)
            for label, value in zip(CLASSIFIER_LABELS, thresholds)
        },
        "candidate_per_role_f1": {
            role: round(score, 6) for role, score in candidate_roles.items()
        },
        "candidate_calibrated_per_role_f1": {
            role: round(score, 6) for role, score in calibrated_roles.items()
        },
        "baseline_per_role_f1": {
            role: round(score, 6) for role, score in baseline_roles.items()
        },
        "per_role_delta": {
            role: round(delta, 6) for role, delta in regressions.items()
        },
        "confusion": confusion_by_role(
            candidate_validation, y[validation], thresholds
        ),
        "augmentation_copies": max(0, args.augment_copies),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "output": str(args.output),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps({
        "schema_version": 1,
        "labels": list(CLASSIFIER_LABELS),
        "features": list(FEATURE_NAMES),
        "thresholds": report["thresholds"],
        "validation": report,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
