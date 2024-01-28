import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "rony/gpt2-quantized-jokes"

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = tokenizer.apply_chat_template('JOKE: ', add_generation_prompt=True, tokenize=False)

# Create pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer
)

# Generate text
sequences = pipeline(
    prompt,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    num_return_sequences=1,
    max_length=200,
)
print(sequences[0]['generated_text'])