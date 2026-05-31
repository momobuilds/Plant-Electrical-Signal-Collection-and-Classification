#!/usr/bin/env python3
"""Create fixed-size 15-second plant-signal windows and statistical features."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import timedelta
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


REQUIRED_RAW_COLUMNS = {
    "timestamp_pc",
    "elapsed_seconds",
    "millis",
    "raw_adc",
    "voltage",
    "lo_plus",
    "lo_minus",
    "label",
    "plant_id",
    "run_id",
}
NUMERIC_COLUMNS = ["elapsed_seconds", "raw_adc", "voltage", "lo_plus", "lo_minus"]
FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "median",
    "rms",
    "energy",
    "skewness",
    "kurtosis",
    "zero_crossing_rate",
    "slope",
    "q25",
    "q75",
    "iqr",
]
WINDOW_METADATA_COLUMNS = [
    "window_id",
    "plant_id",
    "run_id",
    "label",
    "start_time",
    "end_time",
    "start_elapsed_seconds",
    "end_elapsed_seconds",
    "sampling_rate_hz",
    "lead_off_ratio",
    "source_file",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment raw plant recordings and extract statistical features."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing raw CSV files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Folder for generated datasets and plots. Default: data/processed.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help="Target sampling rate in Hz. If omitted, infer the median rate from elapsed_seconds.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=15.0,
        help="Non-overlapping window duration. Default: 15 seconds.",
    )
    parser.add_argument(
        "--signal-column",
        choices=("voltage", "raw_adc"),
        default="voltage",
        help="Signal used for samples and features. Default: voltage.",
    )
    parser.add_argument(
        "--max-lead-off-ratio",
        type=float,
        default=0.10,
        help="Reject windows above this lead-off fraction. Default: 0.10.",
    )
    parser.add_argument(
        "--min-window-coverage",
        type=float,
        default=0.90,
        help="Reject windows with fewer than this fraction of expected samples. Default: 0.90.",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=0.10,
        help="Reject windows containing a larger sampling gap. Default: 0.10 seconds.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {args.input_dir}")
    if args.sampling_rate is not None and args.sampling_rate <= 0:
        raise SystemExit("--sampling-rate must be greater than zero.")
    if args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be greater than zero.")
    if not 0 <= args.max_lead_off_ratio <= 1:
        raise SystemExit("--max-lead-off-ratio must be between 0 and 1.")
    if not 0 < args.min_window_coverage <= 1:
        raise SystemExit("--min-window-coverage must be greater than 0 and at most 1.")
    if args.max_gap_seconds <= 0:
        raise SystemExit("--max-gap-seconds must be greater than zero.")


def infer_sampling_rate(elapsed_seconds: np.ndarray) -> float:
    """Infer Hz from the median positive elapsed-time difference."""
    differences = np.diff(elapsed_seconds)
    positive_differences = differences[differences > 0]
    if positive_differences.size == 0:
        raise ValueError("not enough increasing elapsed_seconds values")
    median_difference = float(np.median(positive_differences))
    if median_difference <= 0 or not math.isfinite(median_difference):
        raise ValueError("invalid elapsed_seconds differences")
    return 1.0 / median_difference


def discover_raw_files(input_dir: Path) -> tuple[list[Path], list[float]]:
    """Find logger CSV files and infer their sampling rates without retaining them."""
    valid_paths: list[Path] = []
    inferred_rates: list[float] = []

    for path in sorted(input_dir.glob("*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            print(f"Skipping unreadable CSV {path.name}: {exc}", file=sys.stderr)
            continue

        missing = REQUIRED_RAW_COLUMNS - set(header.columns)
        if missing:
            print(
                f"Skipping {path.name}: missing raw columns {sorted(missing)}",
                file=sys.stderr,
            )
            continue

        try:
            elapsed = pd.read_csv(path, usecols=["elapsed_seconds"])["elapsed_seconds"]
            elapsed_values = pd.to_numeric(elapsed, errors="coerce").dropna().to_numpy(dtype=float)
            rate = infer_sampling_rate(elapsed_values)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            print(f"Skipping {path.name}: cannot infer sampling rate ({exc})", file=sys.stderr)
            continue

        valid_paths.append(path)
        inferred_rates.append(rate)
        print(f"Found {path.name}: inferred {rate:.3f} Hz")

    return valid_paths, inferred_rates


def finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def extract_features(signal: np.ndarray, sampling_rate: float) -> dict[str, float]:
    """Extract stable summary statistics from one fixed-size signal window."""
    mean_value = float(np.mean(signal))
    centered = signal - mean_value
    q25, q75 = np.percentile(signal, [25, 75])
    time_axis = np.arange(signal.size, dtype=float) / sampling_rate

    if signal.size > 1:
        crossing_rate = float(np.count_nonzero(np.diff(np.signbit(centered))) / (signal.size - 1))
        slope_value = float(np.polyfit(time_axis, signal, 1)[0])
    else:
        crossing_rate = 0.0
        slope_value = 0.0

    return {
        "mean": mean_value,
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "range": float(np.ptp(signal)),
        "median": float(np.median(signal)),
        "rms": float(np.sqrt(np.mean(np.square(signal)))),
        "energy": float(np.sum(np.square(signal))),
        "skewness": finite_or_zero(skew(signal, bias=False)),
        "kurtosis": finite_or_zero(kurtosis(signal, fisher=True, bias=False)),
        "zero_crossing_rate": crossing_rate,
        "slope": slope_value,
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def load_recording(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = pd.read_csv(path)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=NUMERIC_COLUMNS).sort_values("elapsed_seconds")
    if frame.empty:
        raise ValueError("no valid numeric signal rows")

    metadata: dict[str, str] = {}
    for column in ("plant_id", "run_id", "label"):
        values = frame[column].dropna().astype(str).unique()
        if len(values) != 1:
            raise ValueError(f"expected exactly one {column}, found {list(values)}")
        metadata[column] = values[0]
    return frame, metadata


def format_window_time(recording_start: Any, seconds_from_start: float) -> str:
    if pd.isna(recording_start):
        return f"{seconds_from_start:.6f}"
    return (recording_start + timedelta(seconds=seconds_from_start)).isoformat()


def save_example_plot(examples: dict[str, np.ndarray], sampling_rate: float, output_path: Path) -> None:
    labels = sorted(examples)
    figure, axes = plt.subplots(len(labels), 1, figsize=(10, max(3, 2.5 * len(labels))), squeeze=False)
    for axis, label in zip(axes[:, 0], labels):
        signal = examples[label]
        time_axis = np.arange(signal.size, dtype=float) / sampling_rate
        axis.plot(time_axis, signal, linewidth=0.8)
        axis.set_title(label)
        axis.set_ylabel("Signal")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Seconds")
    figure.suptitle("One accepted 15-second plant-signal window per class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths, inferred_rates = discover_raw_files(args.input_dir)
    if not paths:
        raise SystemExit(f"No valid raw logger CSV files found in {args.input_dir}")

    sampling_rate = args.sampling_rate or float(np.median(inferred_rates))
    samples_per_window = int(round(sampling_rate * args.window_seconds))
    if samples_per_window < 2:
        raise SystemExit("The selected sampling rate and window duration produce fewer than two samples.")

    print(
        f"Using target rate {sampling_rate:.3f} Hz and "
        f"{samples_per_window} samples per {args.window_seconds:g}-second window."
    )

    raw_output_path = args.output_dir / "windowed_raw_dataset.csv"
    feature_output_path = args.output_dir / "feature_dataset.csv"
    distribution_output_path = args.output_dir / "class_distribution.csv"
    plot_output_path = args.output_dir / "example_signals_by_class.png"
    sample_columns = [f"sample_{index}" for index in range(samples_per_window)]

    feature_rows: list[dict[str, Any]] = []
    examples: dict[str, np.ndarray] = {}
    class_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()

    with raw_output_path.open("w", newline="", encoding="utf-8") as raw_file:
        raw_writer = csv.DictWriter(raw_file, fieldnames=WINDOW_METADATA_COLUMNS + sample_columns)
        raw_writer.writeheader()

        for path in paths:
            try:
                frame, metadata = load_recording(path)
            except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
                print(f"Skipping {path.name}: {exc}", file=sys.stderr)
                rejected_counts["invalid_file"] += 1
                continue

            elapsed = frame["elapsed_seconds"].to_numpy(dtype=float)
            signal_values = frame[args.signal_column].to_numpy(dtype=float)
            lo_plus = frame["lo_plus"].to_numpy(dtype=float)
            lo_minus = frame["lo_minus"].to_numpy(dtype=float)
            recording_start_elapsed = float(elapsed[0])
            recording_start_pc = pd.to_datetime(frame["timestamp_pc"].iloc[0], errors="coerce")
            relative_duration = float(elapsed[-1] - recording_start_elapsed)
            final_window_index = int(math.floor(relative_duration / args.window_seconds))

            for window_index in range(final_window_index + 1):
                start_elapsed = recording_start_elapsed + window_index * args.window_seconds
                end_elapsed = start_elapsed + args.window_seconds
                mask = (elapsed >= start_elapsed) & (elapsed < end_elapsed)
                window_times = elapsed[mask]
                window_signal = signal_values[mask]
                window_lo_plus = lo_plus[mask]
                window_lo_minus = lo_minus[mask]

                minimum_samples = math.ceil(samples_per_window * args.min_window_coverage)
                if window_signal.size < minimum_samples:
                    rejected_counts["insufficient_samples"] += 1
                    continue

                ideal_times = start_elapsed + np.arange(samples_per_window, dtype=float) / sampling_rate
                gaps = np.diff(window_times)
                maximum_gap = max(
                    float(window_times[0] - ideal_times[0]),
                    float(max(0.0, ideal_times[-1] - window_times[-1])),
                    float(np.max(gaps)) if gaps.size else math.inf,
                )
                if maximum_gap > args.max_gap_seconds:
                    rejected_counts["sampling_gap"] += 1
                    continue

                lead_off_ratio = float(np.mean((window_lo_plus != 0) | (window_lo_minus != 0)))
                if lead_off_ratio > args.max_lead_off_ratio:
                    rejected_counts["lead_off"] += 1
                    continue

                resampled_signal = np.interp(ideal_times, window_times, window_signal)
                window_id = f"{path.stem}_w{window_index:04d}"
                seconds_from_recording_start = window_index * args.window_seconds
                common_row: dict[str, Any] = {
                    "window_id": window_id,
                    "plant_id": metadata["plant_id"],
                    "run_id": metadata["run_id"],
                    "label": metadata["label"],
                    "start_time": format_window_time(recording_start_pc, seconds_from_recording_start),
                    "end_time": format_window_time(
                        recording_start_pc, seconds_from_recording_start + args.window_seconds
                    ),
                    "start_elapsed_seconds": f"{start_elapsed:.6f}",
                    "end_elapsed_seconds": f"{end_elapsed:.6f}",
                    "sampling_rate_hz": f"{sampling_rate:.6f}",
                    "lead_off_ratio": f"{lead_off_ratio:.6f}",
                    "source_file": path.name,
                }
                raw_writer.writerow(
                    common_row
                    | {
                        sample_column: f"{sample_value:.9g}"
                        for sample_column, sample_value in zip(sample_columns, resampled_signal)
                    }
                )
                feature_rows.append(common_row | extract_features(resampled_signal, sampling_rate))
                class_counts[metadata["label"]] += 1
                examples.setdefault(metadata["label"], resampled_signal)

    if not feature_rows:
        raise SystemExit(
            "No windows passed validation. Check recording length, lead-off readings, and sampling gaps."
        )

    pd.DataFrame(feature_rows, columns=WINDOW_METADATA_COLUMNS + FEATURE_NAMES).to_csv(
        feature_output_path, index=False
    )
    distribution = pd.DataFrame(
        sorted(class_counts.items()), columns=["label", "window_count"]
    )
    distribution.to_csv(distribution_output_path, index=False)
    save_example_plot(examples, sampling_rate, plot_output_path)

    print("\nClass distribution:")
    print(distribution.to_string(index=False))
    print(f"\nRejected windows: {dict(rejected_counts) or 'none'}")
    print(f"Saved raw windows: {raw_output_path}")
    print(f"Saved features:    {feature_output_path}")
    print(f"Saved plot:        {plot_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
