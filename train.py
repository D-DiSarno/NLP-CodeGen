from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import logging
import torch
from transformers import AutoTokenizer,EarlyStoppingCallback
import pandas as pd
from datasets import Dataset
from utils import evaluator as e

from transformers import EarlyStoppingCallback
tokenizer = AutoTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-en-nl",
        # "google/flan-t5-base",
        use_fast=True)

batch_size = 1
encoder_length = 32
decoder_length = 32

def map_to_encoder_decoder_inputs(batch):
    inputs = tokenizer(batch["nl"], padding="max_length", truncation=True, max_length=encoder_length)
    outputs = tokenizer(batch["code"], padding="max_length", truncation=True, max_length=decoder_length)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    batch["decoder_attention_mask"] = outputs.attention_mask

    assert all([len(x) == encoder_length for x in inputs.input_ids])
    assert all([len(x) == decoder_length for x in outputs.input_ids])

    return batch

def preprocess(examples, max_length=512):
    model_inputs = tokenizer(examples['nl'],
                             max_length=max_length,
                             padding='max_length',
                             truncation=True,
                             return_attention_mask=True)
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(examples['code'],
                            max_length=max_length,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True)
    model_inputs['labels'] = targets['input_ids']
    model_inputs['decoder_input_ids'] = targets['input_ids']
    model_inputs['decoder_attention_mask'] = targets['attention_mask']
    return model_inputs






def train():
    logging.basicConfig(level=logging.INFO)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_df = pd.read_csv('DJANGO/train.csv')
    val_df = pd.read_csv('DJANGO/val.csv')

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    evaluator = e.CodeGenerationEvaluator(tokenizer, device, smooth_bleu=True)

    train_data = train_dataset.map(
        map_to_encoder_decoder_inputs, batched=True, batch_size=1, remove_columns=['nl', 'code'],
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    # same for validation dataset
    val_data = val_dataset.map(
        map_to_encoder_decoder_inputs, batched=True, batch_size=1, remove_columns=['nl', 'code'],
    )
    val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           max_length=512, padding=True,  ####new
                                           model=model)




    """train_inputs = train_dataset.map(preprocess, batched=True)
    val_inputs = val_dataset.map(preprocess, batched=True)
    train_inputs = train_inputs.remove_columns(['nl', 'code'])
    val_inputs = val_inputs.remove_columns(['nl', 'code'])"""



    training_args = Seq2SeqTrainingArguments(
        output_dir="DJANGO-training",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        num_train_epochs=12,
        do_train=True,
        do_eval=True,
        fp16=False,#set True if cuda
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
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]
    )

    trainer.train()
    trainer.save_model()
