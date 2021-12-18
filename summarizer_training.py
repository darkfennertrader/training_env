import os
import re
import pandas as pd
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

pl.seed_everything(42)


class DialogSum(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        dialogue_max_token_len: int = 512,
        summary_max_token_len: int = 128,
        model_dir: str = "/home/solidsnake/ai/Golden_Group/ai-models/development/summarization/t5-small",
    ):
        self.data = data
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.dialogue_max_len = dialogue_max_token_len
        self.summary_max_len = summary_max_token_len

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
        data_dir: str = "/home/solidsnake/ai/datasets/dialogsum/",
        dialogue_max_token_len: int = 512,
        summary_max_token_len: int = 128,
        batch_size: int = 2,
        n_samples: int = 5,
        model_dir: str = "/home/solidsnake/ai/Golden_Group/ai-models/development/summarization/t5-small",
        view_dataset_stats: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.view_dataset_stats = view_dataset_stats
        self.df = pd.DataFrame()

    def prepare_data(self):
        # print("\nINSIDE PREPARE DATA")
        # print("-" * 60)
        # download, tokenize, ecc...
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
        self.df = DialogSum(df)
        # print(len(self.df))
        # i = 1
        # for elem in self.df:
        #     print(i)
        #     print(elem["dialogue_input_ids"].shape)
        #     print(elem["dialogue_attention_mask"].shape)
        #     print(elem["labels"].shape)
        #     print(elem["labels_attention_mask"].shape)
        #     i += 1

        # print("-" * 60)
        # print()

    def setup(self, stage=None):
        # print("INSIDE SETUP")
        # print("-" * 60)
        # split, transform, ecc...
        train_size, val_size = int(0.8 * len(self.df)), int(0.1 * len(self.df))
        test_size = len(self.df) - (train_size + val_size)
        self.df_train, self.df_val, self.df_test = random_split(
            self.df, [train_size, val_size, test_size]
        )
        print(f"train size, val_size, test_size: {train_size, val_size, test_size}")
        # print("-" * 60)
        # print()

    def train_dataloader(self):
        # print("INSIDE TRAIN DATALOADER")
        # print("-" * 60)
        # print()
        # print("-" * 60)
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


class T5Summarizer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_dir: str = "/home/solidsnake/ai/Golden_Group/ai-models/development/summarization/t5-small"
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_dir, return_dict=True
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
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
        optimizer = AdamW(self.parameters(), eps=1e-4)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=self.args.max_epochs * len(self.train_dat),
        # )
        return {"optimizer": optimizer}


if __name__ == "__main__":

    pretrained_model_name_or_path = (
        "/home/solidsnake/ai/Golden_Group/ai-models/development/summarization/t5-small"
    )
    dataset_dir = "/home/solidsnake/ai/datasets/dialogsum/"

    project = "dialogue-summarizer"
    wandb.init(project=project, dir="logs/")
    # wb_logger = WandbLogger("demo", "wandb/", project=project)
    wandb.finish()
    wb_logger = WandbLogger(project=project)

    # init datamodule
    dm = T5DataModule(batch_size=2)

    # init metrics
    metrics_callback = MetricsCallback()

    # init model
    t5_model = T5Summarizer()

    # init trainer
    trainer = pl.Trainer(
        max_epochs=5,
        enable_progress_bar=True,
        logger=wb_logger,
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


#############################################################################
# dev = pd.read_json(dataset_dir + "dialogsum.dev.jsonl", lines=True, nrows=4)
# dev = dev[["dialogue", "summary"]]
# dev = dev.swifter.applymap(lambda row: re.sub(r" ' ", "'", str(row).replace("\n", " ")))

# train = pd.read_json(dataset_dir + "dialogsum.train.jsonl", lines=True, nrows=4)
# train = train[["dialogue", "summary"]]
# train = train.swifter.applymap(
#     lambda row: re.sub(r" ' ", "'", str(row).replace("\n", " "))
# )

# test = pd.read_json(dataset_dir + "dialogsum.test.jsonl", lines=True, nrows=4)
# test = test[["dialogue", "summary1"]]
# test.rename({"summary1": "summary"}, axis=1, inplace=True)
# test = test.swifter.applymap(
#     lambda row: re.sub(r" ' ", "'", str(row).replace("\n", " "))
# )

# df = pd.concat([dev, train, test])
# df.dropna()

# print(df.head())

# ds = DialogSum(df)
# print(len(ds) == len(df))

# model_dir: str = (
#     "/home/solidsnake/ai/Golden_Group/ai-models/development/summarization/t5-small"
# )
# model = T5ForConditionalGeneration.from_pretrained(model_dir, return_dict=True)


# for item in ds:
# print(item)
# print(item["dialogue"])
# print(item["dialogue_input_ids"])
# print(item["dialogue_input_ids"].shape)
# print(item["dialogue_attention_mask"])
# print(item["dialogue_attention_mask"].shape)
# print(item["labels"])
# print(item["labels"].shape)
# print(item["labels_attention_mask"])
# print(item["labels_attention_mask"].shape)

# input_ids = item["dialogue_input_ids"]
# print(input_ids.shape)
# attention_mask = item["dialogue_attention_mask"]
# labels = item["labels"]
# labels_attention_mask = item["labels_attention_mask"]
# print("############   INFERENCE   #################")
# output = model(
#     input_ids=input_ids,
#     attention_mask=attention_mask,
#     labels=labels,
#     decoder_attention_mask=labels_attention_mask,
# )
# print(output)
# print("#############################################")


##### CHECKING THE DATALOADER ###########

# # init datamodule
# dm = T5DataModule(batch_size=4)
# dm.prepare_data()
# dm.setup()

# print(dm.train_dataloader())

# for batch in dm.train_dataloader():
#     print(f"batch length: {len(batch)}")
#     print(batch)
#     print("\nchecking dimension....")
#     print(
#         batch["dialogue_input_ids"].shape,
#         batch["dialogue_attention_mask"].shape,
#         batch["labels"].shape,
#         batch["labels_attention_mask"].shape,
#     )
#     print()
#     break

####################################################
