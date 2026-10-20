# Data

The original experimental dataset cannot be shared publicly.

## Files used in the thesis

- BI202C01
- BI304C01
- CR202C01
- CR304C01
- FE104C01
- FE202C01
- XXX01C01
- XXX01C03
- XXX01C05

## Material groups

- BI — bismuth films
- CR — chromium films
- FE — iron films

## Reference group

- XXX — modal reference film

The reference film was used to identify conductivity fluctuation components that are not related to the chemical composition of the material films.

## Expected folders

```text
data/
├── q00/
│   ├── BI202C01.Q00
│   ├── BI304C01.Q00
│   ├── CR202C01.Q00
│   ├── CR304C01.Q00
│   ├── FE104C01.Q00
│   ├── FE202C01.Q00
│   ├── XXX01C01.Q00
│   ├── XXX01C03.Q00
│   └── XXX01C05.Q00
└── q03/
    ├── BI202C01.Q03
    ├── BI304C01.Q03
    ├── CR202C01.Q03
    ├── CR304C01.Q03
    ├── FE104C01.Q03
    ├── FE202C01.Q03
    ├── XXX01C01.Q03
    ├── XXX01C03.Q03
    └── XXX01C05.Q03
```

## How to run

```bash
python src/build_dataset.py
python src/train_models.py
python src/train_knn_raw_samples.py
```
