from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
import pandas as pd
import time

def demo_presentazione():
    new_model = AutoModelForSeq2SeqLM.from_pretrained("DJANGO-training/checkpoint-79996")
    new_tokenizer = AutoTokenizer.from_pretrained("DJANGO-training/checkpoint-79996", use_fast=True)
    test_dataset_df = pd.read_csv('DJANGO/test.csv')
    test_dataset = Dataset.from_pandas(test_dataset_df)
    for i in range(len(test_dataset)):
        prompt = test_dataset["nl"][i].strip()

        inputs = new_tokenizer([prompt], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        outputs = new_model.generate(input_ids, attention_mask=attention_mask)
        generated_code = new_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        print(f"Prompt: {prompt}")
        print(f"Predizione del modello: {generated_code}\n")
        time.sleep(5)
