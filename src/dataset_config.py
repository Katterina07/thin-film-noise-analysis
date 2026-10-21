"""Shared dataset metadata. Unknown experimental facts stay unknown."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "q00"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "conductivity_features.csv"
FIGURES_DIR = BASE_DIR / "results" / "figures"
METRICS_DIR = BASE_DIR / "results" / "metrics"
SAMPLE_LENGTH = 128
OVERLAP_MODE = 2
RANDOM_STATE = 42
FEATURES = ["gamma", "log_energy"]
MATERIAL_ORDER = ["BI", "CR", "FE"]
MATERIAL_NAMES = {
    "BI": "Bismuth (Bi)", "CR": "Chromium (Cr)", "FE": "Iron (Fe)",
    "CO": "Cobalt (Co)", "AL": "Aluminum (Al)", "XXX": "Reference film",
}
MATERIAL_COLORS = {
    "BI": "#4C72B0", "CR": "#DD8452", "FE": "#55A868",
    "CO": "#8172B3", "AL": "#937860", "XXX": "#333333",
}
FILE_METADATA = {}


def register(series, material, year, fs, channels, role,
             control_modes=None, measurement_types=None):
    for channel in channels:
        stem = series + channel
        FILE_METADATA[stem] = {
            "material": material,
            "material_name": MATERIAL_NAMES[material],
            "experiment_id": series,
            "channel": channel,
            "year": year,
            "sampling_rate_hz": fs,
            "analysis_role": role,
            "used_for_training": role == "baseline",
            "irradiated": None,
            "control_mode": (control_modes or {}).get(channel, "unknown"),
            "measurement_type": (measurement_types or {}).get(channel, "unknown"),
        }


# baseline/comparison preserve the previous analytical split, not a verified
# claim about physical stability or irradiation. Fill irradiated from the log.
for material, early, late in (
    ("BI", "BI202", "BI304"),
    ("CR", "CR202", "CR304"),
    ("FE", "FE202", "FE104"),
):
    register(early, material, 2005, 1.731449441, ("C01", "C02"), "baseline")
    register(late, material, 2005, 1.731372791, ("C01", "C02"), "baseline")

for material, early, late in (
    ("BI", "BI105", "BI106"),
    ("CR", "CR205", "CR206"),
    ("FE", "FE105", "FE106"),
):
    register(early, material, 2006, 1.733679808, ("C01", "C02"), "comparison")
    register(late, material, 2006, 1.731027741, ("C01", "C02"), "comparison")

register("XXX01", "XXX", 2005, 1.446167970,
         tuple(f"C{i:02d}" for i in range(1, 7)), "reference")
register("CO105", "CO", 2005, 1.728788674, ("C01", "C02"), "exploratory",
         control_modes={"C01": "unregulated", "C02": "regulated"})
register("AL103", "AL", 2004, 1.918461346, ("C01", "C02"), "exploratory",
         control_modes={"C01": "unregulated", "C02": "regulated"})
# year above is the source directory label, not a converted Julian date.
register("COB01", "CO", 2004, 1.446000051,
         ("C01", "C02", "C03"), "exploratory",
         measurement_types={"C01": "total_film_voltage",
                            "C02": "electrode_channel_1",
                            "C03": "electrode_channel_2"})
