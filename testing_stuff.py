from __future__ import annotations

# from data.dataloaders import build_dataloaders
from data.datasets.download import download_all_datasets
from models.hyperparams import ExperimentConfig
from utils.logger import Logger

cfg = ExperimentConfig()

details = Logger()
download_all_datasets(details=details)
