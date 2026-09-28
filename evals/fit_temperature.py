# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fit the serving temperature from saved predictions: uv run evals/fit_temperature.py --help."""

import argparse
import json
import math
from pathlib import Path

from metrics import reliability


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def load_boolq_predictions(path: Path) -> list[tuple[float, float]]:
    """(p_correct, outcome) pairs from a boolq.py run.

    For a binary yes/no question the confidence of the predicted label is
    max(p_yes, 1 - p_yes) with outcome 1 when that label is correct.
    """
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        p_yes = min(max(row["p_yes"], 1e-9), 1 - 1e-9)
        predicted_yes = p_yes >= 0.5
        correct = predicted_yes == row["label"]
        pairs.append((max(p_yes, 1 - p_yes), 1.0 if correct else 0.0))
    return pairs


def load_distributions(path: Path) -> list[tuple[dict[str, float], float]]:
    """Full label distributions from a {probs, correct} jsonl run.

    Cross-checks must keep every label. OPENJEV_TEMPERATURE divides the label
    logits before the softmax, which on a saved distribution is p_i ** (1/T)
    renormalized. Rescaling only logit(max p) is a different transform and
    understates ECE on multi-class questions.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        probs = row.get("probs")
        if not isinstance(probs, dict) or not probs or row.get("correct") is None:
            continue
        rows.append((dict(probs), 1.0 if row["correct"] else 0.0))
    return rows


def top_at(probs: dict[str, float], temperature: float) -> float:
    weights = [max(value, 0.0) ** (1.0 / temperature) for value in probs.values()]
    total = math.fsum(weights)
    if total <= 0:
        raise ValueError("probability mass is not positive")
    return max(weight / total for weight in weights)


def nll(pairs: list[tuple[float, float]], temperature: float) -> float:
    total = 0.0
    for p, y in pairs:
        q = sigmoid(logit(p) / temperature)
        q = min(max(q, 1e-12), 1 - 1e-12)
        total += -(y * math.log(q) + (1 - y) * math.log(1 - q))
    return total / len(pairs)


def ece(pairs: list[tuple[float, float]], temperature: float) -> float:
    scaled = [(sigmoid(logit(p) / temperature), y) for p, y in pairs]
    n = len(scaled)
    return sum(
        g["count"] / n * abs(g["mean_probability"] - g["observed_frequency"])
        for g in reliability([p for p, _ in scaled], [y for _, y in scaled])
        if g["count"]
    )


def ece_distributions(rows: list[tuple[dict[str, float], float]], temperature: float) -> float:
    scaled = [(top_at(probs, temperature), y) for probs, y in rows]
    n = len(scaled)
    return sum(
        g["count"] / n * abs(g["mean_probability"] - g["observed_frequency"])
        for g in reliability([p for p, _ in scaled], [y for _, y in scaled])
        if g["count"]
    )


def fit(pairs: list[tuple[float, float]], lo: float = 0.2, hi: float = 5.0) -> float:
    for _ in range(80):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(pairs, m1) < nll(pairs, m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="boolq.py predictions.jsonl to fit on")
    parser.add_argument(
        "--cross-check",
        type=Path,
        action="append",
        default=[],
        help="extra {probs, correct} jsonl run to validate the fitted temperature on (repeatable)",
    )
    args = parser.parse_args()

    primary = load_boolq_predictions(args.run)
    extras = {path: load_distributions(path) for path in args.cross_check}
    temperature = fit(primary)

    print(f"fit set: {args.run} (n={len(primary)})")
    print(f"fitted temperature (NLL): {temperature:.3f}")
    print("Cross-check ECE uses softmax temperature, p_i ** (1/T), not logit(max p)/T.")
    print()
    header = f"{'T':>6}  {'fit-set ECE':>11}"
    for path in extras:
        header += f"  {path.name + ' ECE':>14}"
    print(header)
    grid = sorted({1.0, round(temperature, 2), 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0})
    for t in grid:
        row = f"{t:>6.2f}  {ece(primary, t):>11.4f}"
        for path, pairs in extras.items():
            row += f"  {ece_distributions(pairs, t):>14.4f}"
        print(row)
    print()
    print("Serving: set OPENJEV_TEMPERATURE in the server environment.")
    print("Caveat: a temperature fitted on one domain can degrade calibration on")
    print("another; prefer a value that improves every cross-check set at once.")


if __name__ == "__main__":
    main()
