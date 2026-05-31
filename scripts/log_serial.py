#!/usr/bin/env python3
"""Log plant electrical signal rows from an Arduino into a labeled CSV file."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import re
import sys
import time
from typing import Sequence

try:
    import serial
except ImportError as exc:  # pragma: no cover - depends on local installation
    raise SystemExit(
        "pyserial is not installed. Run: python3 -m pip install -r requirements.txt"
    ) from exc


OUTPUT_COLUMNS = [
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
]
ARDUINO_HEADER = "millis,raw_adc,voltage,lo_plus,lo_minus"
LEAD_OFF_WARNING_RATIO = 0.10
PROGRESS_INTERVAL_SECONDS = 10.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record labeled plant electrical signals from an Arduino."
    )
    parser.add_argument("--port", required=True, help="Serial port, such as /dev/cu.usbmodem1101.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate. Default: 115200.")
    parser.add_argument("--label", required=True, help="Stimulus label, such as dry or sunlight.")
    parser.add_argument("--plant-id", required=True, help="Plant identifier, such as P1.")
    parser.add_argument("--run-id", required=True, help="Run identifier, such as run1.")
    parser.add_argument(
        "--duration-min",
        type=float,
        required=True,
        help="Recording duration in minutes. Must be greater than zero.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Folder for raw CSV files. Default: data/raw.",
    )
    return parser.parse_args(argv)


def safe_filename_component(value: str) -> str:
    """Keep metadata readable while preventing accidental path creation."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"


def parse_arduino_row(line: str) -> tuple[int, int, float, int, int] | None:
    """Parse and validate one Arduino row. Return None for headers or bad rows."""
    if not line or line == ARDUINO_HEADER:
        return None

    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 5:
        return None

    try:
        millis_value = int(fields[0])
        raw_adc = int(fields[1])
        voltage = float(fields[2])
        lo_plus = int(fields[3])
        lo_minus = int(fields[4])
    except ValueError:
        return None

    if millis_value < 0 or not 0 <= raw_adc <= 1023:
        return None
    if not math.isfinite(voltage) or lo_plus not in (0, 1) or lo_minus not in (0, 1):
        return None

    return millis_value, raw_adc, voltage, lo_plus, lo_minus


def print_progress(
    elapsed: float,
    duration_seconds: float,
    rows_written: int,
    malformed_rows: int,
    lead_off_rows: int,
) -> None:
    lead_off_ratio = lead_off_rows / rows_written if rows_written else 0.0
    print(
        f"[{elapsed:8.1f}/{duration_seconds:.1f}s] "
        f"saved={rows_written} malformed={malformed_rows} "
        f"lead_off={lead_off_rows} ({lead_off_ratio:.1%})",
        flush=True,
    )
    if rows_written >= 100 and lead_off_ratio > LEAD_OFF_WARNING_RATIO:
        print(
            "WARNING: lead-off is high. Check electrode attachment and wiring.",
            file=sys.stderr,
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.baud <= 0:
        raise SystemExit("--baud must be greater than zero.")
    if args.duration_min <= 0:
        raise SystemExit("--duration-min must be greater than zero.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    filename = (
        f"raw_{safe_filename_component(args.plant_id)}_"
        f"{safe_filename_component(args.label)}_"
        f"{safe_filename_component(args.run_id)}_"
        f"{started_at.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    output_path = args.output_dir / filename
    duration_seconds = args.duration_min * 60.0

    rows_written = 0
    malformed_rows = 0
    lead_off_rows = 0
    interrupted = False
    serial_error: str | None = None

    print(f"Writing to {output_path}")
    print("Opening the serial port may reset the Arduino; startup headers are ignored.")

    try:
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            csv_file.flush()

            try:
                with serial.Serial(args.port, args.baud, timeout=1.0) as serial_port:
                    start_monotonic = time.monotonic()
                    next_progress = start_monotonic + PROGRESS_INTERVAL_SECONDS

                    while True:
                        now = time.monotonic()
                        elapsed = now - start_monotonic
                        if elapsed >= duration_seconds:
                            break
                        if now >= next_progress:
                            csv_file.flush()
                            print_progress(
                                elapsed,
                                duration_seconds,
                                rows_written,
                                malformed_rows,
                                lead_off_rows,
                            )
                            next_progress = now + PROGRESS_INTERVAL_SECONDS

                        try:
                            raw_line = serial_port.readline()
                        except serial.SerialException as exc:
                            serial_error = f"Serial read failed: {exc}"
                            break

                        if not raw_line:
                            continue

                        line = raw_line.decode("utf-8", errors="replace").strip()
                        parsed = parse_arduino_row(line)
                        if parsed is None:
                            if line and line != ARDUINO_HEADER:
                                malformed_rows += 1
                            continue

                        millis_value, raw_adc, voltage, lo_plus, lo_minus = parsed
                        now = time.monotonic()
                        elapsed = now - start_monotonic
                        writer.writerow(
                            {
                                "timestamp_pc": datetime.now().astimezone().isoformat(
                                    timespec="milliseconds"
                                ),
                                "elapsed_seconds": f"{elapsed:.6f}",
                                "millis": millis_value,
                                "raw_adc": raw_adc,
                                "voltage": f"{voltage:.6f}",
                                "lo_plus": lo_plus,
                                "lo_minus": lo_minus,
                                "label": args.label,
                                "plant_id": args.plant_id,
                                "run_id": args.run_id,
                            }
                        )
                        rows_written += 1
                        if lo_plus or lo_minus:
                            lead_off_rows += 1
            except serial.SerialException as exc:
                serial_error = f"Could not open serial port {args.port}: {exc}"
            except KeyboardInterrupt:
                interrupted = True
            finally:
                csv_file.flush()
    except OSError as exc:
        raise SystemExit(f"Could not write {output_path}: {exc}") from exc

    lead_off_ratio = lead_off_rows / rows_written if rows_written else 0.0
    status = "Interrupted; partial recording saved." if interrupted else "Recording finished."
    print(status)
    print(
        f"Saved {rows_written} rows to {output_path} "
        f"(malformed={malformed_rows}, lead_off={lead_off_rows}, ratio={lead_off_ratio:.1%})."
    )
    if lead_off_ratio > LEAD_OFF_WARNING_RATIO:
        print(
            "WARNING: more than 10% of saved rows reported lead-off. "
            "Inspect electrode contact before using this recording.",
            file=sys.stderr,
        )
    if serial_error:
        print(f"ERROR: {serial_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
