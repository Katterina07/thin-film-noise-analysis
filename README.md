# Thin-film noise analysis: grouped evaluation

Processing of binary Q00 recordings, spectral feature extraction, and group-aware classification of Bi/Cr/Fe thin-film samples.

This project extends [conductivity-fluctuation-analysis](https://github.com/Katterina07/conductivity-fluctuation-analysis),
where the models were trained and evaluated on the 2005 baseline series (Bi/Cr/Fe) with a random split.
Here the evaluation is grouped by experimental series, and the registry also covers the 2006 series,
the XXX01C01–C06 recordings and the exploratory series CO105, AL103 and COB01 (not used for training). With grouped evaluation the MLP reaches about 53% accuracy, compared with about 88% for a random split.


## Project structure

- `src/dataset_config.py` — central registry of files, channels, and analysis settings.
- `src/signal_processing.py` — reads Q00 recordings and creates processed CSV files.
- `src/train_models.py` — trains and evaluates models, creates summaries and figures.
- `notebooks/01_signal_processing.ipynb` — exploratory signal-processing notebook.
- `notebooks/02_train_models.ipynb` — model-evaluation notebook.

## Registered data

- 24 Bi/Cr/Fe files: two channels for each of 12 experimental series.
- Six `XXX01C01`–`XXX01C06` files are reference recordings and are not automatically included in model training.
- `CO105C01/C02`, `AL103C01/C02`, and `COB01C01/C02/C03` are exploratory recordings.
- Missing registered files are skipped and marked in `file_inventory.csv`, which is written by `src/signal_processing.py`. The notebook checks file availability directly.
- Ag, Pt, and other series are not added automatically because consistent metadata are required.
- Input Q00 files may be placed directly in `data/q00/` or in year-based subfolders.
- Duplicate file names in different subfolders cause an error.

## Labels and limitations

- `baseline` refers to the earlier 2005 Bi/Cr/Fe series and is used as the analytical training base.
- `comparison` refers to the 2006 series, recorded after the films were irradiated.
- `irradiated` comes from the catalogue table. It is `True` for the six comparison series (nanosecond electromagnetic pulses without a carrier frequency on 03-11-2005, 30 minutes for Cr002 and Fe001, 33 minutes for Bi001). It is `False` for the baseline series, which were recorded before that date; this is derived from the recording dates, not stated in the table. It is `None` for the reference and exploratory series.
- `sample_id` is the film number from the table. Baseline films: Bi002, Bi003, Cr002, Cr003, Fe001, Fe002. `CR205/CR206` are the same film (Cr002) as `CR202`, and `FE105/FE106` are the same film (Fe001) as `FE104`, recorded after irradiation. `BI105/BI106` (Bi001) is a different film from both baseline Bi series.
- `used_for_training` preserves the previous analytical selection but does not establish physical stability.
- If this selection is not supported by laboratory records, the evaluation cannot be interpreted as recognition of stable films.
- Channel regimes are taken from the table: `C01` is unregulated and `C02` is regulated for `CO105`, `AL103`, `FE202`, `BI105` and `BI106`; all `COB01` channels are unregulated. For the other Bi/Cr/Fe series the table does not state the regime.
- `year` is a label from the original catalogue table, not a calendar year calculated from Julian Date.
- The table lists start and end dates of each recording (local time). They are not entered in the config. Two end dates (`BI105`, `CR205`) read 2005 and are probably 2006.
- Series of one session (`BI202/CR202/FE202`, `BI304/CR304/FE104`, `BI105/CR205/FE105`, `BI106/CR206/FE106`) share the same start, end and number of points, so they were recorded simultaneously.
- `experiment_id` combines channels belonging to the same experimental series.

## Features

- Window length: 128 points.
- Window overlap: 50%.
- Linear trend removal.
- Blackman window.
- Real FFT (`rFFT`).
- `gamma`: signed slope of `log(power)` versus `log(frequency)`.
- Changing the frequency values without changing the frequency indices used in the fit does not correct the slope.
- `energy`: mean of positive non-zero frequency components of `|FFT|^2`, as in the earlier version of the code. It is not a normalized PSD and not an integrated physical energy.
- `log_energy = log10(energy)`; invalid values are not artificially replaced by a minimum.
- Sampling frequencies in the config are file-level estimates. They agree with the catalogue table at its precision (two decimals); constancy of the sampling step within recordings was not checked.
- The CSV files include `material`, `experiment_id`, `channel`, `record_index`, `sample_start`, `analysis_role`, `control_mode`, `measurement_type`, and `electrode_distance_mm`.
- `sample_start` is the index of the first point of a window within the record, not absolute time.
- Electrode distances for `COB01` are labelled only in `C03`, using record numbers starting from 1.

## Installation and usage

Run the following commands from the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/signal_processing.py
.\.venv\Scripts\python.exe src/train_models.py
.\.venv\Scripts\python.exe -m notebook
```

Open notebooks from the started Jupyter server and select the kernel associated with the installed environment.

By default, the first notebook loads the existing processed CSV file. Set `REBUILD=True` to reconstruct the dataset. If the CSV lacks required columns, the notebook asks for a rebuild.

## Evaluation protocol

- `StratifiedGroupKFold` is used with `experiment_id` as the grouping variable.
- The current configuration uses two folds for the two baseline series of each class.
- Channels from the same experimental series do not appear in both training and test sets.
- If a fold does not contain all classes, the run stops instead of reporting a score.
- `StandardScaler` is fitted within each fold.
- Accuracy is calculated at the window level, not at the level of independent physical samples.
- MLP uses `early_stopping=False`, avoiding a hidden random split of related windows for early stopping.
- After evaluation, the final MLP is trained on all baseline series allowed for training (`used_for_training`).
- Additional materials are not included in the main model.
- Both notebooks call the functions from `train_models.py`. Running them regenerates the figures, and notebook 02 also overwrites the files in `results/metrics`. Older images in `results/figures` are not deleted.

## Results

### Classification models

Two supervised models, KNN and MLP, were evaluated using the extracted spectral features. The MLP scored higher in this evaluation. With two folds, the difference is not statistically assessed.

| Model | Accuracy | 
|---|---:|
| KNN | 43.49% |
| MLP | 52.83% |

Per-fold results are available in `results/metrics/model_metrics.csv`.

### KNN classification report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| BI | 0.377 | 0.766 | 0.505 |
| CR | 0.230 | 0.092 | 0.132 |
| FE | 0.785 | 0.447 | 0.570 |

### MLP classification report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| BI | 0.399 | 0.781 | 0.528 |
| CR | 0.410 | 0.087 | 0.144 |
| FE | 0.864 | 0.716 | 0.783 |

CR is the hardest class for both models (recall 0.092 for KNN and 0.087 for MLP).

### Clustering

KMeans clustering on the training (baseline) data gave a silhouette score of 0.556. For large datasets the score is computed on a subsample. Cluster numbers are not material labels, and the score does not show that the clusters correspond to materials.

## Figures

### Energy and gamma distributions

![Energy boxplots by file](results/figures/energy_boxplots_by_file.png)

![Gamma boxplots by file](results/figures/gamma_boxplots_by_file.png)

### File-level comparison

![File-level energy comparison](results/figures/file_level_energy_comparison.png)

![File-level gamma comparison](results/figures/file_level_gamma_comparison.png)

### Classification results

![KNN confusion matrix](results/figures/knn_confusion_matrix.png)

![MLP confusion matrix](results/figures/mlp_confusion_matrix.png)

Figures are separated by material and channel. Panels use independent y-axis scales; compare axis labels carefully.

Error bars represent the standard deviation across windows within a file, not confidence intervals. Box plots hide outliers only for visualization; the underlying data are not removed.

A high predicted class probability does not prove the absence of a radiation-induced effect.

## Output files

- `data/processed/conductivity_features.csv`
- `data/processed/file_inventory.csv`
- `data/processed/*_features.csv`
- `results/metrics/model_metrics.csv` — per-fold classification metrics.
- `results/metrics/evaluation_splits.csv` — group composition.
- `results/metrics/out_of_fold_predictions.csv`
- `results/metrics/knn_classification_report.csv`
- `results/metrics/mlp_classification_report.csv`
- `results/metrics/clustering_metrics.csv` — silhouette score, reported separately from classification accuracy.
- `results/metrics/comparison_film_predictions.csv` — predictions for all windows in comparison series.
- `results/metrics/comparison_file_summary.csv` — one row per file.
- `results/metrics/irradiated_film_predictions.csv` — only confirmed irradiated rows.
- `results/metrics/file_feature_summary.csv`
- `results/figures/knn_confusion_matrix.png`
- `results/figures/mlp_confusion_matrix.png`
- `results/figures/kmeans_clusters.png`
- `results/figures/file_level_gamma_comparison.png`
- `results/figures/file_level_energy_comparison.png`
- `results/figures/gamma_boxplots_by_file.png`
- `results/figures/energy_boxplots_by_file.png`


## Known limitations

These points have not been verified, so the results are not physical conclusions.

- Independence of series. Within each material the two baseline series come from different films (see `sample_id`), but windows overlap by 50%, and the series of one session were recorded simultaneously, so shared conditions are not excluded. Window-level accuracy may therefore be optimistic. The comparison series `CR205/CR206` and `FE105/FE106` are the same films as `CR202` and `FE104`, so their predictions are not independent of the training data.
- Channel regimes. The table states the regimes only for `FE202`, `BI105`, `BI106`, `CO105`, `AL103` and `COB01`. For the other Bi/Cr/Fe series they are unknown.
- Irradiation status. The status is entered from the catalogue table (see Labels). It is not a model feature, so the metrics above do not change. They were produced before it was entered, so `irradiated_film_predictions.csv` was empty and must be regenerated by a rerun. No conclusions about radiation effects are drawn here.
- Sampling step. Frequencies are computed from the file-level average time step;
  its constancy within a recording was not checked.
- Calibration. Q00 calibration and file structure follow the original processing
  code and were not re-checked.

## Data availability

Raw Q00 files are not included in this repository because they are unpublished
experimental results. Processed features and results are provided to document
the method. To rerun the pipeline, place the files in `data/q00/`.
The MIT license applies to the code only.

Last full run: October 2026.
