# Conductivity Fluctuation Analysis

A Python pipeline for analyzing conductivity fluctuations in thin-film materials using binary `.Q00` measurement files.

The project explores whether spectral features can distinguish bismuth, chromium, and iron films and characterize differences between measurement series and acquisition channels.

## Overview

The pipeline combines signal processing, exploratory analysis, and machine learning:

- Parse binary `.Q00` records and calibration metadata.
- Divide signals into overlapping windows.
- Remove linear trends and extract FFT-based spectral features.
- Preserve measurement-series and channel identifiers.
- Evaluate material classification using grouped cross-validation.
- Compare feature distributions across files and channels.

The primary classification task covers Bi, Cr, and Fe. Additional cobalt, aluminum, and reference measurements are registered for exploratory analysis but are not included in the classifier.

## Dataset

The configuration registers 37 source files:

| Material or group | Measurement series | Channels | Files | Role |
|---|---|---|---:|---|
| Bismuth — Bi | `BI202`, `BI304` | `C01`, `C02` | 4 | Baseline |
| Chromium — Cr | `CR202`, `CR304` | `C01`, `C02` | 4 | Baseline |
| Iron — Fe | `FE202`, `FE104` | `C01`, `C02` | 4 | Baseline |
| Bismuth — Bi | `BI105`, `BI106` | `C01`, `C02` | 4 | Comparison |
| Chromium — Cr | `CR205`, `CR206` | `C01`, `C02` | 4 | Comparison |
| Iron — Fe | `FE105`, `FE106` | `C01`, `C02` | 4 | Comparison |
| Reference | `XXX01` | `C01`–`C06` | 6 | Reference |
| Cobalt — Co | `CO105` | `C01`, `C02` | 2 | Exploratory |
| Cobalt — Co | `COB01` | `C01`–`C03` | 3 | Exploratory |
| Aluminum — Al | `AL103` | `C01`, `C02` | 2 | Exploratory |
| Total | | | 37 | |

Each source file represents a measurement channel, not necessarily an independent physical sample.

### Analytical Roles

- `baseline` — Bi/Cr/Fe series selected as the training dataset.
- `comparison` — Bi/Cr/Fe series analyzed using the trained classifier.
- `reference` — reference measurements excluded from classification.
- `exploratory` — additional measurements reserved for separate investigation.

These roles describe how the pipeline uses the data. They do not establish physical stability or irradiation status.

### Input File Layout

Place source files in `data/q00/`, either directly or in year-based subdirectories:

```text
data/
└── q00/
    ├── 2004/
    │   ├── AL103C01.Q00
    │   ├── AL103C02.Q00
    │   └── ...
    ├── 2005/
    │   ├── BI202C01.Q00
    │   ├── BI202C02.Q00
    │   └── ...
    └── 2006/
        ├── BI105C01.Q00
        ├── BI105C02.Q00
        └── ...
```

Missing registered files are skipped and recorded in `file_inventory.csv`. Duplicate source filenames in different locations trigger an error.

## Signal Processing

Processing is implemented in `src/signal_processing.py`.

For each measurement record, the pipeline:

1. Reads calibration coefficients, metadata, and raw ADC codes.
2. Splits the signal into 128-point windows with 50% overlap.
3. Removes a linear trend from each window.
4. Applies the existing calibration-based scaling.
5. Applies a Blackman window.
6. Computes a real-valued fast Fourier transform (`rFFT`).
7. Extracts spectral features and saves the results.

Windows remain within individual record boundaries.

### Spectral Features

| Feature | Definition |
|---|---|
| `gamma` | Signed slope of `log(power)` versus `log(frequency)` |
| `energy` | Mean positive squared FFT magnitude at nonzero frequencies |
| `log_energy` | Base-10 logarithm of `energy` |

The `gamma` convention is:

\[
\log P(f) = \gamma \log f + C
\]

A decreasing spectrum therefore has a negative slope.

`energy` is an implementation-specific, unnormalized FFT-power feature. It is not a normalized power spectral density or an integrated physical energy.

Sampling frequencies are configured per file using the values provided in the measurement inventory.

### Channel-Aware Metadata

Each feature row preserves:

- Source filename and material label.
- Measurement-series identifier: `experiment_id`.
- Acquisition channel: `channel`.
- Record number: `record_index`.
- Window start index within the record: `sample_start`.
- Sampling frequency and processing parameters.
- Analytical role and available experimental metadata.

For `COB01C03`, electrode separation is assigned using the documented record ranges:

| Record numbers, inclusive | Electrode separation |
|---|---:|
| 1–114 | 16 mm |
| 115–180 | 5 mm |
| 181–238 | 1 mm |
| 239–358 | 10 mm |
| 359–480 | 3 mm |

## Model Evaluation

The classifiers use two input features:

```text
gamma
log_energy
```

### Supervised Models

- K-nearest neighbors — distance-weighted classification baseline.
- Multilayer perceptron — fully connected neural network.

Both models use feature standardization within their training pipelines.

### Grouped Cross-Validation

Evaluation uses `StratifiedGroupKFold` with `experiment_id` as the grouping variable.

All windows and channels belonging to the same measurement series remain together. A series cannot appear in both the training and test partitions of a fold.

The current baseline configuration contains two series per material, resulting in two-fold grouped evaluation.

`StandardScaler` is fitted only on the training partition of each fold. The MLP uses `early_stopping=False` to avoid an additional internal random validation split.

Reported classification metrics include:

- Accuracy.
- Balanced accuracy.
- Precision, recall, and F1-score.
- Confusion matrices based on out-of-fold predictions.

Metrics are calculated at the window level, not at the level of independent physical samples.

### Exploratory Clustering

K-Means provides an exploratory view of the standardized feature space.

Clustering quality is summarized using the silhouette score. Cluster identifiers are not material labels, and the silhouette score is reported separately from classification accuracy.

### Comparison-Series Analysis

After cross-validation, a final MLP is trained on all eligible baseline rows and applied to the comparison series.

The pipeline saves:

- Per-window material predictions.
- Maximum predicted class probabilities.
- File-level feature statistics.
- Majority-vote material predictions.
- Fractions of predictions matching the majority vote and source material label.

The console displays one summary row per file rather than every individual window.

## Installation and Usage

### Requirements

- Python with `venv` and `pip`.
- Git and Git LFS when cloning the repository with its source data.
- Dependencies listed in `requirements.txt`.

### Obtain the Project

Clone the repository using its HTTPS URL, then open PowerShell in the cloned project directory.

Retrieve the source measurement files through Git LFS:

```powershell
git lfs install
git lfs pull
```

### Create the Environment

```powershell
python -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The following commands use the environment's Python executable directly, so activation and changes to PowerShell's execution policy are unnecessary.

### Build the Feature Dataset

```powershell
.\.venv\Scripts\python.exe src/signal_processing.py
```

### Evaluate Models and Generate Plots

```powershell
.\.venv\Scripts\python.exe src/train_models.py
```

### Open the Notebooks

```powershell
.\.venv\Scripts\python.exe -m notebook
```

Run the notebooks in this order:

1. `notebooks/01_signal_processing.ipynb`
2. `notebooks/02_train_models.ipynb`

Use the kernel associated with the project environment.

The first notebook loads an existing feature dataset by default. Set `REBUILD = True` to regenerate it after changing metadata or processing parameters.

Restart notebook kernels after modifying imported Python modules.

## Results and Outputs

The pipeline generates the following output groups:

| Directory | Contents |
|---|---|
| `data/processed/` | Per-file features, combined dataset, and source-file inventory |
| `results/metrics/` | Cross-validation metrics, predictions, and file-level summaries |
| `results/figures/` | Confusion matrices, clustering plots, and feature comparisons |

### Main Data Files

- `conductivity_features.csv` — combined feature dataset.
- `file_inventory.csv` — availability and processing status of registered files.

### Main Evaluation Files

- `model_metrics.csv` — classification metrics for each fold.
- `evaluation_splits.csv` — measurement-series assignments.
- `out_of_fold_predictions.csv` — predictions for held-out windows.
- `comparison_file_summary.csv` — one summary row per comparison file.
- `comparison_film_predictions.csv` — detailed comparison-series predictions.
- `file_feature_summary.csv` — file-level spectral feature statistics.

`irradiated_film_predictions.csv` includes only rows explicitly marked `irradiated=True`. If no irradiation status has been confirmed, the file contains headers but no prediction rows.

### Main Figures

- KNN and MLP confusion matrices.
- Exploratory K-Means clustering.
- File-level mean ± standard deviation comparisons.
- File-level feature-distribution boxplots.

Comparison plots are separated by material and channel. Panels use independent y-axis scales.

Error bars show within-file window standard deviations, not confidence intervals. Boxplots hide outliers only in the visualization; the underlying data are retained.

Images from previous runs are not automatically deleted.

## Project Structure

```text
.
├── data/
│   ├── q00/
│   └── processed/
├── notebooks/
│   ├── 01_signal_processing.ipynb
│   └── 02_train_models.ipynb
├── results/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── dataset_config.py
│   ├── signal_processing.py
│   └── train_models.py
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

File metadata and shared settings are maintained in `src/dataset_config.py`.

The notebooks reuse functions from the Python modules rather than maintaining separate processing and training implementations.

## Limitations and Experimental Metadata

The following limitations must be considered when interpreting results:

- Irradiation status remains unknown until verified against the experimental log. Baseline and comparison roles must not be treated as confirmed irradiation labels.
- Bi/Cr/Fe `C01` and `C02` represent different channels, but their regulation modes have not been established.
- For `CO105` and `AL103`, the documented channel modes are `C01 = unregulated` and `C02 = regulated`.
- Measurement series are used as validation groups, but independence of physical samples must be verified. Multiple series from one sample require a broader grouping identifier.
- Synchronized acquisition sessions across materials may require additional session-based validation.
- Sampling frequencies are file-level average estimates. Uniform timing within records must be checked.
- The configured `year` field is a source-directory label, not a calendar year converted from Julian dates.
- Calibration assumptions and the binary Q00 structure must be verified against instrument documentation.
- The spectral slope is fitted across all usable nonzero FFT frequencies, not a separately validated 1/f frequency band.
- A high predicted class probability does not demonstrate the absence of irradiation-induced or other physical changes.

## Validation Status

The Python scripts and notebook code cells have been checked for syntax.

End-to-end execution on the complete source dataset and scientific validation of the resulting models remain to be verified.

No numerical performance claims are made in this README.

## Data Availability and Licensing

The repository is configured to track source `.Q00` files using Git LFS.

Generated feature datasets and large per-window prediction tables are excluded from version control and can be regenerated using the processing scripts.

Source-data redistribution permissions and the licensing terms for code and measurements must be confirmed and documented before public release.
