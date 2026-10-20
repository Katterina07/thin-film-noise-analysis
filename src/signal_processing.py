"""Read Q00 files and extract the existing FFT features per channel.

energy is mean unnormalized squared FFT magnitude, not a calibrated PSD.
gamma is the signed log(power)/log(frequency) slope, not its negation.
"""
import struct
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.signal.windows import blackman
from dataset_config import (
    DATA_DIR, PROCESSED_DIR, DATASET_PATH, FILE_METADATA,
    SAMPLE_LENGTH, OVERLAP_MODE,
)

FEATURE_COLUMNS = ["gamma", "energy", "jd1", "jd2", "record_index", "sample_start"]


def read_q00_file(file_path):
    file_path = Path(file_path)
    with file_path.open("rb") as stream:
        calibration = stream.read(8)
        if len(calibration) != 8:
            raise ValueError(f"{file_path.name}: truncated calibration header")
        km, ku = struct.unpack("<ff", calibration)
        record_index = 0
        while True:
            header = stream.read(32)
            if not header:
                break
            record_index += 1
            if len(header) != 32:
                raise ValueError(f"{file_path.name}: truncated record {record_index}")
            jd1, jd2, up1, up2, kc, n_values = struct.unpack("<ddffii", header)
            if n_values <= 0:
                raise ValueError(f"{file_path.name}: invalid n_values in record {record_index}")
            payload = stream.read(n_values * 2)
            if len(payload) != n_values * 2:
                raise ValueError(f"{file_path.name}: truncated samples in record {record_index}")
            yield dict(km=km, ku=ku, jd1=jd1, jd2=jd2, up1=up1, up2=up2,
                       kc=kc, codes=np.frombuffer(payload, dtype="<i2"))


def sample_step(sample_length=SAMPLE_LENGTH, overlap_mode=OVERLAP_MODE):
    if not isinstance(sample_length, (int, np.integer)) or sample_length < 4:
        raise ValueError("sample_length must be an integer >= 4")
    if overlap_mode not in (0, 1, 2, 3):
        raise ValueError("overlap_mode must be 0, 1, 2 or 3")
    return max(1, int(sample_length * (1, .75, .5, .25)[overlap_mode]))


def iter_samples(codes, sample_length=SAMPLE_LENGTH, overlap_mode=OVERLAP_MODE):
    """Yield (start index within record, sample); never cross record boundaries."""
    step = sample_step(sample_length, overlap_mode)
    for start in range(0, len(codes) - sample_length + 1, step):
        yield start, codes[start:start + sample_length]


def split_into_samples(codes, sample_length=SAMPLE_LENGTH, overlap_mode=OVERLAP_MODE):
    """Compatibility interface: return samples only, as in the previous version."""
    return [sample for _, sample in iter_samples(codes, sample_length, overlap_mode)]


def compute_spectrum_features(sample, sampling_rate):
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive and finite")
    sample = np.asarray(sample, dtype=float)
    if sample.ndim != 1 or len(sample) < 4 or not np.isfinite(sample).all():
        return np.nan, np.nan
    power = np.abs(rfft(sample * blackman(len(sample)))) ** 2
    frequencies = np.fft.rfftfreq(len(sample), d=1 / sampling_rate)
    valid = (frequencies > 0) & (power > 0) & np.isfinite(power)
    if valid.sum() < 2:
        return np.nan, np.nan
    gamma, _ = np.polyfit(np.log(frequencies[valid]), np.log(power[valid]), 1)
    return float(gamma), float(power[valid].mean())


def process_q00_file(file_path, sample_length=SAMPLE_LENGTH,
                     overlap_mode=OVERLAP_MODE, sampling_rate=1.73):
    sample_step(sample_length, overlap_mode)
    features = []
    n_records = 0
    time_index = np.arange(sample_length)
    for record_index, record in enumerate(read_q00_file(file_path), start=1):
        n_records += 1
        denominator = (record["kc"] * record["ku"] * record["km"]
                       * (record["up1"] + record["up2"]))
        if not np.isfinite(denominator) or denominator == 0:
            raise ValueError(f"{Path(file_path).name}: invalid scale, record {record_index}")
        for start, sample in iter_samples(record["codes"], sample_length, overlap_mode):
            sample = np.asarray(sample, dtype=float)
            trend = np.polyval(np.polyfit(time_index, sample, 1), time_index)
            detrended = (sample - trend) * (2.0 / denominator)
            gamma, energy = compute_spectrum_features(detrended, sampling_rate)
            features.append(dict(gamma=gamma, energy=energy,
                                 jd1=record["jd1"], jd2=record["jd2"],
                                 record_index=record_index, sample_start=start))
    if not n_records:
        raise ValueError(f"No valid records in {Path(file_path).name}")
    return pd.DataFrame(features, columns=FEATURE_COLUMNS)


def electrode_distance_mm(record_index):
    for first, last, distance in (
        (1, 114, 16.), (115, 180, 5.), (181, 238, 1.),
        (239, 358, 10.), (359, 480, 3.),
    ):
        if first <= record_index <= last:
            return distance
    return np.nan


def find_input_file(input_dir, file_stem):
    """Support files directly in q00 or in year subdirectories."""
    matches = [p for p in Path(input_dir).rglob("*")
               if p.is_file() and p.name.upper() == f"{file_stem}.Q00".upper()]
    if len(matches) > 1:
        raise ValueError(f"Duplicate source files for {file_stem}: {matches}")
    return matches[0] if matches else None


def process_file(input_dir, file_stem, output_dir,
                 sample_length=SAMPLE_LENGTH, overlap_mode=OVERLAP_MODE,
                 sampling_rate=None):
    metadata = FILE_METADATA[file_stem]
    input_path = find_input_file(input_dir, file_stem)
    if input_path is None:
        raise FileNotFoundError(f"Missing {file_stem}.Q00 in {input_dir}")
    fs = metadata["sampling_rate_hz"] if sampling_rate is None else sampling_rate
    df = process_q00_file(input_path, sample_length, overlap_mode, fs)
    df["file"] = file_stem
    for column, value in metadata.items():
        df[column] = value
    df["sampling_rate_hz"] = fs
    df["sample_length"] = sample_length
    df["overlap_mode"] = overlap_mode
    df["log_energy"] = np.nan
    valid_energy = np.isfinite(df["energy"]) & (df["energy"] > 0)
    df.loc[valid_energy, "log_energy"] = np.log10(df.loc[valid_energy, "energy"])
    df["electrode_distance_mm"] = np.nan
    if file_stem == "COB01C03":
        df["electrode_distance_mm"] = df["record_index"].map(electrode_distance_mm)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{file_stem}_features.csv", index=False)
    print(f"{file_stem}: {len(df)} windows")
    return df


def build_combined_dataset():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames, inventory = [], []
    for stem, metadata in FILE_METADATA.items():
        path = find_input_file(DATA_DIR, stem)
        if path is None:
            inventory.append(dict(file=stem, status="missing", n_samples=0, **metadata))
            print(f"Missing, skipped: {stem}.Q00")
            continue
        frame = process_file(DATA_DIR, stem, PROCESSED_DIR)
        frames.append(frame)
        inventory.append(dict(file=stem, status="processed", n_samples=len(frame), **metadata))
    pd.DataFrame(inventory).to_csv(PROCESSED_DIR / "file_inventory.csv", index=False)
    if not frames or not any(len(frame) for frame in frames):
        raise RuntimeError("No complete windows found; check input files and sample length")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(DATASET_PATH, index=False)
    print(f"Saved {len(combined)} windows from {combined['file'].nunique()} files: {DATASET_PATH}")
    return combined


def main():
    build_combined_dataset()


if __name__ == "__main__":
    main()
