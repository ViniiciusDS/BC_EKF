from pathlib import Path
import csv

from src.odometry import (
    EncoderConfig,
    DifferentialDriveConfig,
    build_dataset_from_encoder_and_uwb,
    load_and_validate_encoder_file,
)

BASE_DIR = Path(__file__).resolve().parent




encoder_file = BASE_DIR / "encoder_square.csv"
uwb_file = BASE_DIR / "uwb_square.csv"

cfg = DifferentialDriveConfig(
    wheel_radius_m=0.03,
    wheel_base_m=0.16,
    encoder=EncoderConfig(ticks_per_wheel_rev=600.0),
)

encoder_samples = load_and_validate_encoder_file(encoder_file)

with uwb_file.open("r", encoding="utf-8-sig", newline="") as f:
    uwb_rows = list(csv.DictReader(f))

dataset = build_dataset_from_encoder_and_uwb(
    encoder_samples,
    uwb_rows,
    cfg,
)

print(dataset["aligned_rows"][:5])