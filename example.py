	from transformers import GPT2LMHeadModel, GPT2Tokenizer

text = input()

model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
input_ids = tokenizer.encode(text, return_tensors="pt")
out = model.generate(input_ids)
generated_text = list(map(tokenizer.decode, out))[0]
print(generated_text)

