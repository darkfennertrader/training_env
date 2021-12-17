# PyTorch Lightning integration with Ray Tune
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining


from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback


def tune_asha(num_samples=10, gpus_per_trial=1, data_dir="/ray_tune"):

    config = {
        # define search space
        "lr": tune.loguniform(1e-6, 1e-4),
        "num_epochs": tune.choice([3, 4, 5]),
    }

    scheduler = ASHAScheduler(max_t=5, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["val_loss"],
    )

    train_fn_with_parameters = tune.with_parameters(
        "func that contains pl.trainer()",
        num_gpus=gpus_per_trial,
        data_dir=data_dir,
        batch_size=2,
    )

    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial={"cpu": os.cpu_count(), "gpu": gpus_per_trial},
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_T5_asha",
        callbacks=[
            WandbLoggerCallback(
                project="Optimization_Project",
                api_key="b68430b76b30624cf20fa75f717e8b58f873c4d0",
                log_config=True,
            )
        ],
    )

    print("Best hyperparameters found were: ", analysis.best_config)
