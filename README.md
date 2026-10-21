# Conductivity Fluctuation Analysis

Processing of binary Q00 recordings, spectral feature extraction, and group-aware classification of Bi/Cr/Fe thin-film samples.

## Project structure

- `src/dataset_config.py` — central registry of files, channels, and analysis settings.
- `src/signal_processing.py` — reads Q00 recordings and creates processed CSV files.
- `src/train_models.py` — trains and evaluates models, creates summaries and figures.
- `notebooks/01_signal_processing.ipynb` — exploratory signal-processing notebook.
- `notebooks/02_train_models.ipynb` — model-evaluation notebook.
- `build_dataset.py` is unused and is not required.

## Registered data

- 24 Bi/Cr/Fe files: two channels for each of 12 experimental series.
- Six `XXX01C01`–`XXX01C06` files are reference recordings and are not automatically included in model training.
- `CO105C01/C02`, `AL103C01/C02`, and `COB01C01/C02/C03` are exploratory recordings.
- Missing registered files are skipped and marked in `file_inventory.csv`.
- Ag, Pt, and other series are not added automatically because consistent metadata are required.
- Input Q00 files may be placed directly in `data/q00/` or in year-based subfolders.
- Duplicate file names in different subfolders cause an error.

## Labels and limitations

- `baseline` refers to the earlier 2005 Bi/Cr/Fe series and is used as the analytical training base.
- `comparison` refers to the earlier 2006 series. This does not establish an irradiation status.
- `irradiated=None` until confirmed from laboratory records; set `True` or `False` only based on documented evidence.
- `used_for_training` preserves the previous analytical selection but does not establish physical stability.
- If this selection is not supported by laboratory records, the evaluation cannot be interpreted as recognition of stable films.
- `C01` and `C02` for Bi/Cr/Fe are treated as different channels; their operating regimes are unknown.
- For `CO105` and `AL103`, regimes are taken from the provided description: `C01` is unregulated and `C02` is regulated.
- `year` is a label from the original catalogue table, not a calendar year calculated from Julian Date.
- Dates from earlier responses were not transferred because they were not reliably verified.
- `experiment_id` combines channels belonging to the same experimental series. The independence of series and physical samples requires verification.
- If several series belong to the same physical sample, a broader group identifier is required.
- Synchr