from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def test():

    new_model = AutoModelForSeq2SeqLM.from_pretrained("DJANGO-training/checkpoint-79996")
    new_tokenizer = AutoTokenizer.from_pretrained("DJANGO-training/checkpoint-79996", use_fast=True)
    while True:
        prompt = input("Prompt: ")

        inputs = new_tokenizer([prompt], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        outputs = new_model.generate(input_ids, attention_mask=attention_mask)
        generated_code = new_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print(generated_code)
