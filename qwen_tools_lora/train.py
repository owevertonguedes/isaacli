"""Treina LoRA no Qwen2.5-Coder-3B pra emitir <tool_call>. Roda na T4 do Colab.

Mede ANTES e DEPOIS no mesmo conjunto de teste — prompts que NAO estao no treino.
A metrica e binaria e mecanica: a saida tem <tool_call> com JSON valido dentro?
"""
import json
import re
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

BASE = "Qwen/Qwen2.5-Coder-3B-Instruct"

# Prompts de TESTE — nenhum aparece no treino. Sem isso mediriamos decoreba.
TESTE = [
    ("Leia o arquivo relatorio.txt", "read_file"),
    ("apaga e reescreve o placar.json com o texto vazio", "write_file"),
    ("quais arquivos estao na pasta build?", "list_dir"),
    ("roda ai o comando: git diff --stat", "run_command"),
    ("tem alguma alteracao pendente no repo?", "git_status"),
    ("cola 'fim do documento' no final do texto.md", "append_file"),
    ("me diz o que tem escrito no arquivo licenca.txt", "read_file"),
    ("cria um arquivo teste.py escrevendo print(1)", "write_file"),
]

SYS_TESTE = open("sys_teste.txt").read() if __import__("os").path.exists("sys_teste.txt") else None


def gerar(modelo, tok, pedido, sistema):
    msgs = [{"role": "system", "content": sistema}, {"role": "user", "content": pedido}]
    entrada = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(entrada, return_tensors="pt").to(modelo.device)
    # autocast: com a lm_head treinada ela fica em fp32 (exigencia do GradScaler)
    # enquanto os hidden states chegam em fp16 -> erro de dtype no matmul.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = modelo.generate(**ids, max_new_tokens=120, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    # skip_special_tokens=False e OBRIGATORIO: <tool_call> e token ESPECIAL
    # adicionado (151657). Com skip=True ele some da string e a regra de
    # avaliacao daria 0/8 mesmo com o modelo acertando tudo — falso negativo.
    bruto = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    return bruto.replace("<|im_end|>", "").replace("<|endoftext|>", "")


def avaliar(modelo, tok, sistema, rotulo):
    acertos, detalhes = 0, []
    for pedido, esperado in TESTE:
        saida = gerar(modelo, tok, pedido, sistema)
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", saida, re.S)
        ok = False
        if m:
            try:
                ok = json.loads(m.group(1)).get("name") == esperado
            except json.JSONDecodeError:
                ok = False
        acertos += ok
        detalhes.append((pedido[:40], ok, saida.strip()[:90].replace("\n", " ")))
    print(f"\n===== {rotulo}: {acertos}/{len(TESTE)} com <tool_call> correto =====")
    for p, ok, s in detalhes:
        print(f"  [{'OK ' if ok else 'ERR'}] {p:42s} -> {s}")
    return acertos


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token

    linhas = [json.loads(l) for l in open("treino.jsonl")]
    # Reaproveita o system prompt do proprio dataset pro teste ser comparavel.
    sistema = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", linhas[0]["text"], re.S).group(1)

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16)
    modelo = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=q,
                                                  device_map="auto", torch_dtype=torch.float16)

    antes = avaliar(modelo, tok, sistema, "ANTES do LoRA")

    modelo.gradient_checkpointing_enable()
    modelo.enable_input_require_grads()
    # A lm_head e quem escolhe o token de saida. <tool_call> e o token unico
    # 151657, que o modelo base praticamente nunca emite: sem treinar a lm_head
    # o LoRA aprende o CONTEUDO (o JSON sai no formato certo) e nunca a TAG.
    # Medido: loss travava em 0.713 com grad_norm 0.003 (saturacao). Com a
    # lm_head dentro, o mesmo teste caiu pra 1e-06 e a tag saiu.
    modelo = get_peft_model(modelo, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["lm_head"]))
    # Parametro treinavel TEM que ser fp32: o GradScaler do fp16 recusa
    # desescalar gradiente fp16 ("Attempting to unscale FP16 gradients").
    for p in modelo.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    modelo.print_trainable_parameters()

    MARCA = "<|im_start|>assistant\n"

    def tokenizar(b):
        """Treina SO na resposta do assistente.

        Duas coisas quebraram antes: (1) o DataCollatorForLanguageModeling
        sobrescreve qualquer `labels` que a gente monte aqui — por isso ele foi
        removido em favor do collator padrao; (2) treinar no exemplo inteiro faz
        o alvo real (a tag <tool_call>, ~5% dos tokens) se perder no meio do
        system prompt gigante. Mascarando o prompto com -100, todo o gradiente
        vai pro que a gente quer ensinar.
        """
        r = tok(b["text"], truncation=True, max_length=1024, padding="max_length")
        labels = []
        for texto, ids, mask in zip(b["text"], r["input_ids"], r["attention_mask"]):
            corte = len(tok(texto[: texto.index(MARCA) + len(MARCA)])["input_ids"])
            labels.append([
                (t if (m == 1 and i >= corte) else -100)
                for i, (t, m) in enumerate(zip(ids, mask))
            ])
        r["labels"] = labels
        return r

    ds = Dataset.from_list(linhas).map(tokenizar, batched=True, remove_columns=["text"])

    Trainer(
        model=modelo, train_dataset=ds,
        args=TrainingArguments(
            output_dir="./saida", num_train_epochs=8, per_device_train_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=1e-4, fp16=True,
            logging_steps=10, save_strategy="no", report_to=[], optim="paged_adamw_8bit"),
    ).train()

    modelo.save_pretrained("./lora_tool_call")
    modelo.eval()
    depois = avaliar(modelo, tok, sistema, "DEPOIS do LoRA")

    print(f"\n########## RESULTADO: {antes}/{len(TESTE)} -> {depois}/{len(TESTE)} ##########")
    json.dump({"antes": antes, "depois": depois, "total": len(TESTE)},
              open("resultado.json", "w"))


if __name__ == "__main__":
    main()
