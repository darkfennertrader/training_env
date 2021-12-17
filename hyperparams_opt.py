# PyTorch Lightning integration with Ray Tune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)

tune_report_callback = TuneReportCallback({"loss": "ptl/val_loss"}, on="validation_end")
