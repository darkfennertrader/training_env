import os
import timeit
from functools import partial
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn
import torchvision
from transformers.convert_graph_to_onnx import convert
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5Tokenizer,
    BertTokenizer,
    AutoConfig,
    BartForConditionalGeneration,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    GPT2Config,
    GPT2DoubleHeadsModel,
    GPT2Tokenizer,
)

save_dir = (
    "/home/solidsnake/ai/Golden_Group/ai-models/development/summarization/t5-small"
)
model_name_or_path = "t5-small"


# Natural Language Intent (Zero-Shot Classification)
config = AutoConfig.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(
    model_name_or_path,
    from_tf=False,
    config=config,
)

config.save_pretrained(save_dir)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)


# instantiate from dir
# tokenizer = AutoTokenizer.from_pretrained(save_dir)
# model = T5ForConditionalGeneration.from_pretrained(save_dir)
