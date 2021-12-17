from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
model_checkpoint = ModelCheckpoint(
    dirpath="checkpoints", monitor="val_loss", mode="min", save_top_k=1, verbose=True
)


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)
