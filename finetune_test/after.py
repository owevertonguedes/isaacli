import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
PROMPT = "Escreva um script C# para Unity de um controle de personagem 2D simples para um jogo mobile (iOS/Android): movimento lateral com toque na tela, e um pulo com física básica usando Rigidbody2D. Comente as partes principais."

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Carregando modelo base + adaptador LoRA...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="cuda")
model = PeftModel.from_pretrained(base, "./lora_adapter")

messages = [{"role": "user", "content": PROMPT}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")

print("Gerando resposta DEPOIS do fine-tune...", flush=True)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=500, temperature=0.2, do_sample=True)
text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

with open("output_after.txt", "w") as f:
    f.write(text)

print("=== DEPOIS ===")
print(text)
