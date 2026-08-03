"""Onde exatamente a mascara de labels cai, em tokens. Sem GPU, segundos.

Hipotese: `corte = len(tok(prompt))` desalinha na fronteira, porque tokenizar o
prompt sozinho nao da os mesmos ids que tokenizar prompt+resposta juntos (o
tokenizer funde tokens na emenda). Se desalinhar, o primeiro token supervisionado
nao e o da tag <tool_call> — e o modelo nunca aprende a emiti-la, exatamente o
sintoma medido (lixo no lugar da tag, JSON certo depois).
"""
import json

from transformers import AutoTokenizer

BASE = "Qwen/Qwen2.5-Coder-3B-Instruct"
MARCA = "<|im_start|>assistant\n"

tok = AutoTokenizer.from_pretrained(BASE)
texto = json.loads(open("treino.jsonl").readline())["text"]
prompt = texto[: texto.index(MARCA) + len(MARCA)]

inteiro = tok(texto)["input_ids"]
so_prompt = tok(prompt)["input_ids"]
corte = len(so_prompt)

print(f"tokens do texto inteiro : {len(inteiro)}")
print(f"tokens so do prompt     : {corte}")
print(f"prefixo bate?           : {inteiro[:corte] == so_prompt}")
if inteiro[:corte] != so_prompt:
    for i, (a, b) in enumerate(zip(inteiro, so_prompt)):
        if a != b:
            print(f"  DIVERGE no indice {i}: inteiro={a}({tok.decode([a])!r}) "
                  f"prompt={b}({tok.decode([b])!r})")
            break

print("\nprimeiros 8 tokens SUPERVISIONADOS (o que o LoRA tenta aprender):")
for t in inteiro[corte:corte + 8]:
    print(f"  {t:>7}  {tok.decode([t])!r}")

print("\ncomo <tool_call> e tokenizado sozinho:")
ids_tag = tok("<tool_call>", add_special_tokens=False)["input_ids"]
print(f"  {ids_tag} -> {[tok.decode([i]) for i in ids_tag]}")
print(f"  e token especial adicionado? {'<tool_call>' in tok.get_added_vocab()}")
print(f"  added_vocab relevantes: "
      f"{ {k: v for k, v in tok.get_added_vocab().items() if 'tool' in k} }")

print("\nARMADILHA: decode com skip_special_tokens=True apaga token especial.")
print(f"  skip=False -> {tok.decode(ids_tag, skip_special_tokens=False)!r}")
print(f"  skip=True  -> {tok.decode(ids_tag, skip_special_tokens=True)!r}")
