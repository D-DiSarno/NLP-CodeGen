from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import logging
import torch
from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset
from utils import evaluator as e

tokenizer = AutoTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-en-nl",
        # "google/flan-t5-base",
        use_fast=True)


def preprocess(examples, max_length=512):
    model_inputs = tokenizer(examples['intent'],
                             max_length=max_length,
                             padding='max_length',
                             truncation=True,
                             return_attention_mask=True)
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(examples['snippet'],
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

    train_df = pd.read_csv('CoNaLa-Large/train.csv')
    val_df = pd.read_csv('CoNaLa-Large/val.csv')

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    evaluator = e.CodeGenerationEvaluator(tokenizer, device, smooth_bleu=True)

    batch_size = 1

    train_inputs = train_dataset.map(preprocess, batched=True)
    val_inputs = val_dataset.map(preprocess, batched=True)
    train_inputs = train_inputs.remove_columns(['intent', 'snippet'])
    val_inputs = val_inputs.remove_columns(['intent', 'snippet'])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           max_length=512,
                                           model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="Fine-Tuned-Models",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        num_train_epochs=5,
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
