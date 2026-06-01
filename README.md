# Plant Electrical Signal Collection and Classification

This project records plant electrical signals from an AD8232 module and an
Arduino UNO, Nano ESP32, builds non-overlapping 15-second
samples, extracts statistical features, and trains baseline machine-learning
classifiers.

The workflow is inspired by the 2025 paper
[Classifying plant electrical signals in response to external stimuli using machine learning for enhanced agricultural sustainability](https://www.tandfonline.com/doi/full/10.1080/27685241.2025.2534470).
The paper collected AD8232 and Arduino plant signals for five conditions and
segmented recordings into labeled 15-second samples.

## Safety Scope

This project is for plant measurements only. Do not connect this setup or its
electrodes to a person. The AD8232 is commonly sold as an ECG front end, but
this repository is deliberately limited to plant electrophysiology.

## Project Structure

```text
.
├── arduino/
│   └── plant_signal_acquisition/
│       └── plant_signal_acquisition.ino
├── scripts/
│   ├── log_serial.py
│   ├── generate_synthetic_data.py
│   ├── create_dataset.py
│   └── train_models.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic_raw/
│   └── synthetic_processed/
├── results/
├── requirements.txt
└── README.md
```

## Wiring

The five module connections match the
[SparkFun AD8232 hookup guide](https://learn.sparkfun.com/tutorials/ad8232-heart-rate-monitor-hookup-guide/all):

| AD8232 pin | Arduino UNO pin |
| --- | --- |
| `GND` | `GND` |
| `3.3V` | `3.3V` |
| `OUTPUT` | `A0` |
| `LO+` | `D10` |
| `LO-` | `D11` |

`LO+` and `LO-` are lead-off indicators. A high value means the module reports
an electrode-contact problem. The logger records these values, warns when they
are frequent, and the dataset builder rejects windows with excessive lead-off
readings.

Keep electrode placement consistent across recordings. The referenced paper
attached electrodes to plant leaves and recorded the watered condition
30 minutes after watering, rather than while pouring water.

### Nano Boards

For a classic ATmega328P Nano or Nano ESP32, use the same labeled pins:
`3V3`, `GND`, `A0`, `D10`, and `D11`. Do not power the AD8232 from `5V` or
`VBUS`.

The sketch detects Nano ESP32 builds automatically. The Nano ESP32 does not
provide the AVR `analogReference()` API, so the sketch uses its 12-bit ADC and
the Arduino-ESP32 calibrated millivolt conversion. Select **By Arduino pin
(default)** under **Tools > Pin Numbering** when using a Nano ESP32.

The classic Nano follows the UNO's AVR path: `DEFAULT` is nominally 5 V and
`analogRead()` returns values from `0` through `1023`.

## ADC Reference Voltage

With the five wires above, the sketch uses:

```cpp
analogReference(DEFAULT);
```

On the 5 V Arduino UNO, `DEFAULT` is nominally 5 V. The fact that the AD8232 is
powered from the UNO `3.3V` pin does **not** change the ADC reference. The
default sketch therefore calculates:

```text
voltage = raw_adc * 5.0 / 1024.0
```

The UNO ADC is 10-bit, so `analogRead()` returns `0` through `1023`. USB supply
variation means the calculated voltage is approximate unless the actual
reference voltage is measured.

The sketch also supports an optional 3.3 V external ADC reference for finer
resolution:

1. Disconnect power.
2. Add a wire from Arduino `3.3V` to Arduino `AREF`.
3. Set `USE_EXTERNAL_3V3_AREF = true` in the sketch.
4. Recompile and upload the sketch.

That mode uses `analogReference(EXTERNAL)` and:

```text
voltage = raw_adc * 3.3 / 1024.0
```

Do not connect `3.3V` to `AREF` while the sketch uses `DEFAULT`. Arduino's
[`analogReference()` documentation](https://docs.arduino.cc/language-reference/en/functions/analog-io/analogReference/)
warns that mixing an external AREF voltage with the active internal reference
can damage the microcontroller. The AD8232 itself is a 2.0 V to 3.5 V
single-supply device according to the
[Analog Devices AD8232 datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/AD8232.pdf).

## Arduino Upload

1. Open
   `arduino/plant_signal_acquisition/plant_signal_acquisition.ino`
   in the Arduino IDE.
2. Select your exact Arduino board and the correct serial port.
3. Upload the sketch.

The sketch samples at 100 Hz, uses a rollover-safe `micros()` scheduler, and
prints this header at startup:

```text
millis,raw_adc,voltage,lo_plus,lo_minus
```

Each following row is one sample. Serial speed is `115200` baud.

## Python Installation

Use Python 3.10 or newer. From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`uv` is also supported:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Collect Raw Recordings

Find the Arduino port after uploading. On macOS it commonly looks like
`/dev/cu.usbmodem1101`. Replace that example port in every command if needed:

```bash
ls /dev/cu.usbmodem* /dev/cu.usbserial*
```

Record each requested condition for 30 minutes:

```bash
python scripts/log_serial.py --port /dev/cu.usbmodem1101 --baud 115200 --label dry --plant-id P1 --run-id run1 --duration-min 30 --output-dir data/raw
python scripts/log_serial.py --port /dev/cu.usbmodem1101 --baud 115200 --label sunlight --plant-id P1 --run-id run1 --duration-min 30 --output-dir data/raw

python scripts/log_serial.py --port /dev/cu.usbmodem1101 --baud 115200 --label watered --plant-id P1 --run-id run1 --duration-min 30 --output-dir data/raw
python scripts/log_serial.py --port /dev/cu.usbmodem1101 --baud 115200 --label cutting --plant-id P1 --run-id run1 --duration-min 30 --output-dir data/raw
```

For the `watered` class, apply water first and wait 30 minutes if reproducing
the paper's protocol. For repeated recordings, change `run1` to `run2`, `run3`,
and so on. Keep the same label spelling across runs.

The logger:

- Writes files such as `raw_P1_dry_run1_20260531_143000.csv`.
- Adds PC timestamps, elapsed seconds, labels, plant IDs, and run IDs.
- Reports progress every 10 seconds.
- Ignores malformed serial rows without stopping.
- Flushes data periodically and preserves partial recordings after `Ctrl+C`.
- Warns if more than 10% of saved rows report lead-off.

## Generate Synthetic Demo Data

Create logger-compatible synthetic recordings when you want to test the
pipeline without waiting for real plant measurements:

```bash
python scripts/generate_synthetic_data.py
```

The default command writes one 3-minute recording for each label to
`data/synthetic_raw`. These generated files are deliberately separated from
`data/raw` and must not be treated as experimental results.

Build and train against the synthetic demo data:

```bash
python scripts/create_dataset.py --input-dir data/synthetic_raw --output-dir data/synthetic_processed --sampling-rate 100
python scripts/train_models.py --input-csv data/synthetic_processed/feature_dataset.csv --output-dir results/synthetic_demo
```

Use `--overwrite` to regenerate existing synthetic files. Additional options
can change the duration, number of runs, random seed, and injected lead-off
ratio:

```bash
python scripts/generate_synthetic_data.py --runs-per-label 2 --duration-min 5 --lead-off-ratio 0.02 --overwrite
```

## Build Datasets

Create fixed 15-second samples at the intended 100 Hz rate:

```bash
python scripts/create_dataset.py --input-dir data/raw --output-dir data/processed --sampling-rate 100
```

To infer the target sampling rate from `elapsed_seconds`, omit
`--sampling-rate 100`.

The builder creates:

```text
data/processed/windowed_raw_dataset.csv
data/processed/feature_dataset.csv
data/processed/class_distribution.csv
data/processed/example_signals_by_class.png
```

Accepted windows are resampled onto a fixed time grid so every raw row contains
the same `sample_0`, `sample_1`, ... columns. By default, the builder rejects
partial windows, windows with a sampling gap above 0.10 seconds, and windows
with more than 10% lead-off rows.

Features are:

```text
mean, std, min, max, range, median, rms, energy, skewness, kurtosis,
zero_crossing_rate, slope, q25, q75, iqr
```

`zero_crossing_rate` is calculated after subtracting the window mean because
the AD8232 output is biased above zero.

## Train Baseline Models

Train Random Forest, Extra Trees, and KNN models:

```bash
python scripts/train_models.py --input-csv data/processed/feature_dataset.csv --output-dir results
```

The script uses a stratified 70/30 train-test split and requests stratified
10-fold cross-validation. If the smallest class contains fewer than 10
windows, it prints a warning and reduces the fold count so training can still
run. It saves:

```text
results/results_metrics.csv
results/confusion_matrix_random_forest.png
results/confusion_matrix_extra_trees.png
results/confusion_matrix_knn.png
results/classification_report_random_forest.csv
results/classification_report_extra_trees.csv
results/classification_report_knn.csv
```

The reported precision, recall, and F1 values are macro averages across
classes. A random window split is useful for a baseline comparison, but it can
overestimate real-world performance when windows from one recording appear in
both partitions. For stronger evaluation, collect multiple plants and runs and
later evaluate held-out plants or held-out runs.

## Useful Options

```bash
python scripts/log_serial.py --help
python scripts/generate_synthetic_data.py --help
python scripts/create_dataset.py --help
python scripts/train_models.py --help
```

The dataset builder uses voltage by default. Add `--signal-column raw_adc` to
extract samples and features from ADC counts instead.
