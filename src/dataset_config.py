"""Shared dataset metadata. Unknown experimental facts stay unknown.

Source of the per-file facts below: the catalogue table (Реестр.xlsx).
"""
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

# Channel regimes stated in the table (channel 1 unregulated, channel 2 regulated).
REGULATION = {"C01": "unregulated", "C02": "regulated"}
# Irradiation stated in the table: nanosecond electromagnetic pulses without a
# carrier frequency, 03-11-2005 (30 min for Cr002 and Fe001, 33 min for Bi001).
IRRADIATION_DATE = "2005-11-03"


def register(series, material, year, fs, channels, role,
             control_modes=None, measurement_types=None,
             sample_id=None, irradiated=None, irradiation_date=None):
    for channel in channels:
        stem = series + channel
        FILE_METADATA[stem] = {
            "material": material,
            "material_name": MATERIAL_NAMES[material],
            "experiment_id": series,
            "sample_id": sample_id,
            "channel": channel,
            "year": year,
            "sampling_rate_hz": fs,
            "analysis_role": role,
            "used_for_training": role == "baseline",
            "irradiated": irradiated,
            "irradiation_date": irradiation_date,
            "control_mode": (control_modes or {}).get(channel, "unknown"),
            "measurement_type": (measurement_types or {}).get(channel, "unknown"),
        }


# Baseline: 2005 recordings made before the 03-11-2005 irradiation, so
# irradiated=False means "not irradiated at the time of recording". This is
# derived from the recording dates; the table does not state it directly.
# Sample numbers come from the table. Bi, Cr and Fe series of one session
# (same start, end and number of points) were recorded simultaneously.
BASELINE = (
    # series, material, sample, fs, control modes
    ("BI202", "BI", "Bi002", 1.731449441, None),
    ("CR202", "CR", "Cr002", 1.731449441, None),
    ("FE202", "FE", "Fe002", 1.731449441, REGULATION),
    ("BI304", "BI", "Bi003", 1.731372791, None),
    ("CR304", "CR", "Cr003", 1.731372791, None),
    ("FE104", "FE", "Fe001", 1.731372791, None),
)
for series, material, sample, fs, modes in BASELINE:
    register(series, material, 2005, fs, ("C01", "C02"), "baseline",
             control_modes=modes, sample_id=sample, irradiated=False)

# Comparison: 2006 recordings of films irradiated on 03-11-2005 (table).
# CR205/CR206 are the same film as CR202 (Cr002), and FE105/FE106 are the same
# film as FE104 (Fe001), now after irradiation. BI105/BI106 (Bi001) was not
# recorded before irradiation.
COMPARISON = (
    ("BI105", "BI", "Bi001", 1.733679808, REGULATION),
    ("BI106", "BI", "Bi001", 1.731027741, REGULATION),
    ("CR205", "CR", "Cr002", 1.733679808, None),
    ("CR206", "CR", "Cr002", 1.731027741, None),
    ("FE105", "FE", "Fe001", 1.733679808, None),
    ("FE106", "FE", "Fe001", 1.731027741, None),
)
for series, material, sample, fs, modes in COMPARISON:
    register(series, material, 2006, fs, ("C01", "C02"), "comparison",
             control_modes=modes, sample_id=sample, irradiated=True,
             irradiation_date=IRRADIATION_DATE)

register("XXX01", "XXX", 2005, 1.446167970,
         tuple(f"C{i:02d}" for i in range(1, 7)), "reference")
# CO105 was recorded 18-10-2005 to 24-11-2005, which spans 03-11-2005. The table
# does not say that Co001 was irradiated, so irradiated stays None.
register("CO105", "CO", 2005, 1.728788674, ("C01", "C02"), "exploratory",
         control_modes=REGULATION, sample_id="Co001")
# Table sample number is 1, the description text says Al000.
register("AL103", "AL", 2005, 1.918461346, ("C01", "C02"), "exploratory",
         control_modes=REGULATION, sample_id="Al000")
# year above is the "Год" column of the table, not a converted Julian date.
# The table states that all COB01 channels are unregulated.
register("COB01", "CO", 2004, 1.446000051,
         ("C01", "C02", "C03"), "exploratory",
         control_modes={c: "unregulated" for c in ("C01", "C02", "C03")},
         measurement_types={"C01": "total_film_voltage",
                            "C02": "electrode_channel_1",
                            "C03": "electrode_channel_2"},
         sample_id="Co001")