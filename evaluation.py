from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import evaluator as e
import torch
import pandas as pd
from datasets import Dataset

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')  # Use CPU
new_tokenizer = AutoTokenizer.from_pretrained("DJANGO-training/checkpoint-12", use_fast=True)
evaluator = e.CodeGenerationEvaluator(new_tokenizer, device, smooth_bleu=True)
new_model = AutoModelForSeq2SeqLM.from_pretrained("DJANGO-training/checkpoint-12")


# map data correctly
def generate_new_code(batch):
    inputs = new_tokenizer(batch["nl"], padding="max_length", truncation=True, return_tensors="pt")

    # max_new_tokens=512,

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    outputs = new_model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = new_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred_code"] = output_str

    return batch



def eval():

    batch_size = 1
    test_dataset_df = pd.read_csv('DJANGO/test.csv')
    test_dataset = Dataset.from_pandas(test_dataset_df)
    results = test_dataset.map(generate_new_code, batched=True, batch_size=batch_size)

    # Ok Evaluator
    bleu_score = 0
    rouge_score = 0
    sacre_bleu = 0

    for i in range(len(results)):
        ref = results["code"][i].strip()
        pred = results["pred_code"][i].replace('▁', ' ').strip()
        if pred is not None and pred != "":
            if ref is not None and ref != "":
                # bleu_metric = metric.compute(predictions=[pred], references=[[ref]])
                # bleu_score  += bleu_metric["score"]

                bleu_metric = evaluator.evaluate([pred], [ref])
                # bleu_metric = evaluator.evaluateSingle(pred, ref)

                # bleu_score += bleu_metric['BLEU-Unigram-Precision']
                bleu_score += bleu_metric['BLEU']
                rouge_score += bleu_metric['ROUGE-L']
                sacre_bleu += bleu_metric['SacreBLEU']
                # print(bleu_metric['SacreBLEU'])
        else:
            continue
    print('Bleu Score: {} (scale 0-100)'.format(bleu_score / len(results)))

    print('Sacre Bleu: {} (scale 0-100)'.format(sacre_bleu / len(results)))

    print('ROUGE Score: {} (scale 0-100)'.format(rouge_score / len(results)))