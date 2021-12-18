import os
import re
import pandas as pd
import argparse
from argparse import ArgumentParser
import swifter
from rouge_score import rouge_scorer
from torch.utils.data import Dataset, DataLoader, random_split
from torch import cuda
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from callback_collections import (
    MyPrintingCallback,
    MetricsCallback,
    early_stopping,
    model_checkpoint,
)

# from hyperparams_opt import tune_report_callback

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    get_linear_schedule_with_warmup,
)

from tqdm.auto import tqdm
import wandb


# from ray.tune.integration.wandb import WandbLogger

from pprint import pprint

pd.set_option("display.max_colwidth", 80)

##############################################################
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

sns.set(style="whitegrid", palette="muted", font_scale=1.2)
rcParams["figure.figsize"] = 16, 10
##############################################################


class DialogSum(Dataset):
    def __init__(
        self,
        args,
        data: pd.DataFrame,
    ):
        self.args = args
        self.data = data
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.model)
        self.dialogue_max_len = self.args.dialogue_max_token_len
        self.summary_max_len = self.args.summary_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        dialogue = row["dialogue"]
        dialogue = str(re.sub(r" ' ", "'", dialogue))
        dialogue = " ".join(dialogue.split())
        dialogue_encoding = self.tokenizer(
            dialogue,
            max_length=self.dialogue_max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        summary = row["summary"]
        summary = str(re.sub(r" ' ", "'", summary))
        summary = " ".join(summary.split())
        summary_encoding = self.tokenizer(
            summary,
            max_length=self.summary_max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            dialogue=dialogue,
            summary=summary,
            dialogue_input_ids=dialogue_encoding["input_ids"].flatten(),
            dialogue_attention_mask=dialogue_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten(),
        )


class T5DataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        view_dataset_stats: bool = False,
    ):
        super().__init__()
        self.args = args
        self.data_dir = self.args.data_dir
        self.batch_size = self.args.batch_size
        self.n_samples = self.args.n_samples
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.model)
        self.view_dataset_stats = view_dataset_stats
        self.df = pd.DataFrame()

    def prepare_data(self):
        train = pd.read_json(
            self.data_dir + "dialogsum.train.jsonl", lines=True, nrows=self.n_samples
        )
        train = train[["dialogue", "summary"]]
        train = train.swifter.applymap(
            lambda row: re.sub(r" ' ", "'", str(row).replace("\n", " "))
        )

        dev = pd.read_json(
            self.data_dir + "dialogsum.dev.jsonl", lines=True, nrows=self.n_samples
        )
        dev = dev[["dialogue", "summary"]]
        dev = dev.swifter.applymap(
            lambda row: re.sub(r" ' ", "'", str(row).replace("\n", " "))
        )

        test = pd.read_json(
            self.data_dir + "dialogsum.test.jsonl", lines=True, nrows=self.n_samples
        )
        test = test[["dialogue", "summary1"]]
        test.rename({"summary1": "summary"}, axis=1, inplace=True)
        test = test.swifter.applymap(
            lambda row: re.sub(r" ' ", "'", str(row).replace("\n", " "))
        )

        df = pd.concat([train, dev, test])
        df.dropna()

        print(df.head())
        print(f"\ndataset shape: {df.shape}")

        if self.view_dataset_stats:
            # dataset statistics
            dialogue_token_counts, summary_token_counts = [], []

            for _, row in df.iterrows():
                dialogue_token_count = len(self.tokenizer.encode(row["dialogue"]))
                dialogue_token_counts.append(dialogue_token_count)
                summary_token_count = len(self.tokenizer.encode(row["summary"]))
                summary_token_counts.append(summary_token_count)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            sns.histplot(dialogue_token_counts, ax=ax1)
            ax1.set_title("dialogue token counts")
            sns.histplot(summary_token_counts, ax=ax2)
            ax2.set_title("summary token counts")
            plt.show()

        # return tokenized dict
        self.df = DialogSum(self.args, df)

    def setup(self, stage=None):
        train_size, val_size = int(0.8 * len(self.df)), int(0.1 * len(self.df))
        test_size = len(self.df) - (train_size + val_size)
        self.df_train, self.df_val, self.df_test = random_split(
            self.df, [train_size, val_size, test_size]
        )
        print(f"train size, val_size, test_size: {train_size, val_size, test_size}")

    def train_dataloader(self):
        return DataLoader(
            self.df_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.df_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.df_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=False,
        )

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # add dataset specific arguments HERE:

        parser.add_argument(
            "--dialogue_max_token_len",
            type=int,
            default=512,
        )
        parser.add_argument(
            "--summary_max_token_len",
            type=int,
            default=128,
        )
        parser.add_argument(
            "--n_samples",
            type=int,
            default=5,
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default="/home/solidsnake/ai/datasets/dialogsum/",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=2,
        )
        return parser


class T5Summarizer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.args.model, return_dict=True
        )
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.model)
        self.learning_rate = self.args.learning_rate
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        # print("inside forward step")
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        # print("inside training step")
        loss, _ = self.shared_step(batch, train=True)
        self.log(
            "training_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["dialogue_input_ids"]),
            logger=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # print("inside validation step")
        loss, _ = self.shared_step(batch, train=False)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        print(outputs)
        loss = sum([o["loss"] for o in outputs]) / len(outputs)
        out = {"val_loss": loss}
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {**out, "log": out}

    def test_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, train=False)
        return {"loss": loss}

    # def test_epoch_end(self, outputs):
    #     loss = sum([o["loss"] for o in outputs]) / len(outputs)
    #     out = {"test_loss": loss}
    #     return {**out, "log": out}

    def shared_step(self, batch, train):
        input_ids = batch["dialogue_input_ids"]
        attention_mask = batch["dialogue_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        # useful for logging different metrics depending on the phase: (training, validation, test)
        if train:
            pass
        else:
            pass

        return loss, outputs

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), eps=self.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=self.args.max_epochs * len(self.train_dat),
        # )
        return {"optimizer": optimizer}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # add model specific arguments HERE:
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default="/home/solidsnake/ai/Golden_Group/ai-models/development/summarization/t5-small",
            help="name of the model or the path pointing to it",
        )
        parser.add_argument(
            "-lr",
            "--learning_rate",
            type=float,
            default=1e-4,
        )
        return parser


def parse_arguments():
    p = ArgumentParser()
    p = T5Summarizer.add_model_specific_args(p)
    p = T5DataModule.add_dataset_specific_args(p)
    args, _ = p.parse_known_args()
    return args


def main():

    args = parse_arguments()

    project = "dialogue-summarizer"
    # wandb.init(project=project)
    # wandb.finish()
    # wb_logger = WandbLogger(project=project)

    pl.seed_everything(42)

    # init datamodule
    dm = T5DataModule(args)

    # init metrics
    metrics_callback = MetricsCallback()

    # init model
    t5_model = T5Summarizer(args)

    # init trainer
    trainer = pl.Trainer(
        max_epochs=2,
        enable_progress_bar=True,
        logger=False,  # wb_logger,
        gpus=cuda.device_count(),
        # precision=16,
        num_sanity_val_steps=2,
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[metrics_callback, model_checkpoint],
    )

    # train the model
    trainer.fit(t5_model, dm)

    # optionally: run test
    # trainer.test()


if __name__ == "__main__":
    main()
