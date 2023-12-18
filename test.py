from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def test():
    new_model = AutoModelForSeq2SeqLM.from_pretrained("./Marian-Training")
    new_model.to("cuda")

    new_tokenizer = AutoTokenizer.from_pretrained("./Marian-Training", use_fast=True)

    def generate_code(prompt):
        inputs = new_tokenizer([prompt], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        outputs = new_model.generate(input_ids, attention_mask=attention_mask)

        output_code = new_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_code

    natural_language = input()
    gen_code = generate_code(natural_language)

    print(gen_code)
