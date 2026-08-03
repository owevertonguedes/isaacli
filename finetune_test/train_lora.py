import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataset import build_examples

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="cuda")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

examples = build_examples()

def tokenize(example):
    prompt_ids = tok.apply_chat_template(
        [example["messages"][0]], add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"][0]
    full_ids = tok.apply_chat_template(
        example["messages"], add_generation_prompt=False, return_tensors="pt", return_dict=True
    )["input_ids"][0]
    labels = full_ids.clone()
    labels[: len(prompt_ids)] = -100
    return {"input_ids": full_ids, "labels": labels, "attention_mask": torch.ones_like(full_ids)}

tokenized = [tokenize(ex) for ex in examples]

def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), max_len), tok.pad_token_id or tok.eos_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        l = len(x["input_ids"])
        input_ids[i, :l] = x["input_ids"]
        labels[i, :l] = x["labels"]
        attn[i, :l] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

args = TrainingArguments(
    output_dir="./lora_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=12,
    learning_rate=2e-4,
    logging_steps=2,
    save_strategy="no",
    bf16=True,
    report_to=[],
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collate)
trainer.train()

model.save_pretrained("./lora_adapter")
print("LoRA salvo em ./lora_adapter")
