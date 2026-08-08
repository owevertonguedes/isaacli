"""Teste de sanidade do pipeline LoRA: o modelo consegue DECORAR 1 exemplo?

Um modelo que nao decora um unico exemplo com 50 epochs e lr alto tem bug de
engenharia, nao limite de capacidade. Esse teste separa as duas hipoteses da
task 01 em ~2 min de GPU, em vez de mais um treino completo as cegas.

Fases (todas no mesmo processo, mesmo objeto de modelo carregado):
  A) baseline sem adaptador nenhum  -> prova que o modelo base fala
  B) overfit 1 exemplo, gerar com adaptador ATIVO
  C) mesmo objeto, disable_adapter() -> isola adaptador do resto do pipeline

Se B ficar mudo e C falar, o problema e o adaptador (ou a fusao dele).
Se B e C ficarem mudos, o problema e o estado do modelo apos o Trainer
(quantizacao/fp16/gradient checkpointing), nao o LoRA.
"""
import json
import re
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

BASE = "Qwen/Qwen2.5-Coder-3B-Instruct"
SEM_QUANT = "--sem-quant" in sys.argv   # hipotese fp16+4bit instavel na T4
SEM_FP16 = "--sem-fp16" in sys.argv     # hipotese fp16 no Trainer estoura
LM_HEAD = "--lm-head" in sys.argv       # treina MLP + lm_head (rota do token novo)

MARCA = "<|im_start|>assistant\n"


def gerar(modelo, tok, texto_prompt, rotulo):
    ids = tok(texto_prompt, return_tensors="pt").to(modelo.device)
    # autocast e obrigatorio quando a lm_head foi treinada: ela fica em fp32
    # (exigencia do GradScaler) enquanto os hidden states chegam em fp16 —
    # sem isso da "expected mat1 and mat2 to have the same dtype: Half != float".
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = modelo.generate(**ids, max_new_tokens=80, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    novos = out[0][ids["input_ids"].shape[1]:]
    bruto = tok.decode(novos, skip_special_tokens=False)
    limpo = tok.decode(novos, skip_special_tokens=True)
    print(f"\n----- {rotulo} -----")
    print(f"  tokens gerados : {len(novos)}  ids={novos[:8].tolist()}")
    print(f"  bruto          : {bruto[:220]!r}")
    print(f"  limpo          : {limpo.strip()[:220]!r}")
    print(f"  MUDO           : {'SIM' if len(limpo.strip()) == 0 else 'nao'}")
    return limpo.strip()


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token

    linhas = [json.loads(l) for l in open("treino.jsonl")]
    exemplo = linhas[0]["text"]
    prompt = exemplo[: exemplo.index(MARCA) + len(MARCA)]
    alvo = exemplo[exemplo.index(MARCA) + len(MARCA):]
    print(f"exemplo unico. alvo esperado (inicio): {alvo.strip()[:160]!r}")

    carga = dict(device_map="auto", dtype=torch.float16)
    if not SEM_QUANT:
        carga["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    print(f"\n=== carregando (quantizado={not SEM_QUANT}, fp16_trainer={not SEM_FP16}) ===")
    modelo = AutoModelForCausalLM.from_pretrained(BASE, **carga)

    gerar(modelo, tok, prompt, "A) BASE, sem adaptador (controle)")

    modelo.gradient_checkpointing_enable()
    modelo.enable_input_require_grads()
    # target_modules so de atencao NAO consegue ensinar um token novo: quem
    # decide qual token sai e a lm_head. <tool_call> e o id unico 151657, que o
    # base quase nunca emite — sem lm_head treinavel o gradiente satura (medido:
    # loss trava em 0.713, grad_norm -> 0.003) e sai lixo no lugar da tag.
    alvos = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if LM_HEAD:
        alvos += ["gate_proj", "up_proj", "down_proj"]
    modelo = get_peft_model(modelo, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=alvos,
        modules_to_save=["lm_head"] if LM_HEAD else None))
    # Parametro treinavel TEM que ser fp32: o GradScaler do fp16 recusa
    # desescalar gradiente fp16 ("Attempting to unscale FP16 gradients").
    # O resto do modelo segue quantizado — so o que recebe gradiente sobe.
    for p in modelo.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    modelo.print_trainable_parameters()

    def tokenizar(b):
        r = tok(b["text"], truncation=True, max_length=1024)
        corte = len(tok(prompt)["input_ids"])
        r["labels"] = [[(t if i >= corte else -100) for i, t in enumerate(ids)]
                       for ids in r["input_ids"]]
        return r

    # 1 exemplo repetido: da passos de otimizador suficientes pra decorar.
    ds = Dataset.from_list([linhas[0]] * 8).map(tokenizar, batched=True,
                                                remove_columns=["text"])

    Trainer(
        model=modelo, train_dataset=ds,
        args=TrainingArguments(
            output_dir="./saida_sanidade", num_train_epochs=50,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            learning_rate=2e-4, fp16=not SEM_FP16, logging_steps=25,
            save_strategy="no", report_to=[], optim="paged_adamw_8bit"),
    ).train()

    modelo.eval()
    saida_com = gerar(modelo, tok, prompt, "B) DEPOIS do overfit, adaptador ATIVO")
    with modelo.disable_adapter():
        saida_sem = gerar(modelo, tok, prompt, "C) mesmo objeto, adaptador DESLIGADO")

    decorou = bool(re.search(r"<tool_call>", saida_com))
    print("\n" + "#" * 60)
    print(f"# DECOROU o exemplo (tem <tool_call>) : {'SIM' if decorou else 'NAO'}")
    print(f"# adaptador ATIVO mudo                : {not saida_com}")
    print(f"# adaptador DESLIGADO mudo            : {not saida_sem}")
    if decorou:
        print("# => PIPELINE SAO. Problema e dado/hiperparametro no treino completo.")
    elif saida_sem and not saida_com:
        print("# => O ADAPTADOR e o culpado (base fala com ele desligado).")
    else:
        print("# => Estado do modelo pos-Trainer quebrado (quant/fp16), nao o LoRA.")
    print("#" * 60)


if __name__ == "__main__":
    main()
