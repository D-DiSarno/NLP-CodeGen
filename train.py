import transformers
import datasets
from datasets import load_dataset, load_metric
import logging
from transformers import BertTokenizer, GPT2Tokenizer, GPT2TokenizerFast, EncoderDecoderModel, Trainer, TrainingArguments, BertTokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import types
import argparse
import logging
from functools import partial
import json

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertGenerationConfig,
    BertGenerationEncoder,
    BertTokenizer,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    AutoTokenizer
)

import sacrebleu
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from utils import evaluator as E


def train():
    def preprocess(examples, MAX_LENGTH=512):
        model_inputs = tokenizer(examples['intent'],
                                 max_length=MAX_LENGTH,
                                 padding='max_length',
                                 truncation=True,
                                 return_attention_mask=True)
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(examples['snippet'],
                                max_length=MAX_LENGTH,
                                padding='max_length',
                                truncation=True,
                                return_attention_mask=True)
        model_inputs['labels'] = targets['input_ids']
        model_inputs['decoder_input_ids'] = targets['input_ids']
        model_inputs['decoder_attention_mask'] = targets['attention_mask']
        return model_inputs

    logging.basicConfig(level=logging.INFO)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    raw_dataset = datasets.load_dataset(
        'csv',
        data_files={
            'train': './CoNaLa/train.csv',
            'validation': './CoNaLa/val.csv',
            'test': './CoNaLa/test.csv'},
        # delimeter=',',
        quotechar='"')

    train_df = pd.read_csv('CoNaLa/train.csv')
    val_df = pd.read_csv('CoNaLa/val.csv')
    test_df = pd.read_csv('CoNaLa/test.csv')

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-en-nl",
        use_fast=True)

    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
    evaluator = E.CodeGenerationEvaluator(tokenizer, device, smooth_bleu=True)

    max_length = 512
    batch_size = 1
    learning_rate = 1e-5

    train_inputs = train_dataset.map(preprocess, batched=True)
    val_inputs = val_dataset.map(preprocess, batched=True)
    train_inputs = train_inputs.remove_columns(['intent', 'snippet'])
    val_inputs = val_inputs.remove_columns(['intent', 'snippet'])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           max_length=512,
                                           model=model)

    train_dataloader = torch.utils.data.DataLoader(train_inputs,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=data_collator)
    val_dataloader = torch.utils.data.DataLoader(val_inputs,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 collate_fn=data_collator)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./Marian-Training",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        num_train_epochs=3,
        do_train=True,
        do_eval=True,
        fp16=True,
        overwrite_output_dir=True,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        seed=1995,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=evaluator,
        data_collator=data_collator,
        train_dataset=train_inputs,
        eval_dataset=val_inputs,
    )

    trainer.train()
    trainer.save_model()
    # trainer.save_state()
