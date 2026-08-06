# Experimento LoRA no Qwen2.5-Coder-3B — CONCLUÍDO, deu certo

**Não é código ativo.** É a prova de um experimento encerrado em 2026-07-19.
O app não usa mais o Qwen (usa `granite4:micro-h`). Mantido porque documenta a
resposta a uma pergunta que custou dias.

## A pergunta

Um bom gerador de código **aprende** a emitir `<tool_call>` e criar arquivos,
mexendo nos pesos via LoRA?

## A resposta: SIM

```
ANTES do LoRA:  0/8    ```json { "name": "read_file", ... }
DEPOIS do LoRA: 6/8    <tool_call>{"name":"read_file",...}</tool_call>
```

Os 8 prompts de teste **não estão** no `treino.jsonl` — é generalização.
Dos 2 "erros", ambos eram gabarito errado (ver defeito conhecido abaixo).
Nos itens respondíveis: **6/6**.

## A causa raiz (o que destravou)

`<tool_call>` é o token único **151657**. Quem escolhe o token de saída é a
`lm_head` — e ela estava fora do `target_modules`. O adaptador aprendia o
CONTEÚDO (o JSON saía no formato compacto exato do dataset) e nunca a TAG.

Assinatura do problema: **loss travada em 0.713 com grad_norm → 0.003**. Isso é
saturação, não convergência. Se vir esse padrão de novo, procure uma camada
necessária que ficou fora do treino.

Correção: `modules_to_save=["lm_head"]` + MLP (`gate_proj`, `up_proj`, `down_proj`)
nos `target_modules`. Loss foi pra **1e-06**.

## Duas armadilhas da correção (custaram duas rodadas)

1. `ValueError: Attempting to unscale FP16 gradients` — parâmetro treinável tem
   que ser **fp32**: `p.data = p.data.float()`. O GradScaler recusa desescalar
   gradiente fp16.
2. `expected mat1 and mat2 to have the same dtype: c10::Half != float` na
   geração — a `lm_head` fica em fp32 (exigência acima) e os hidden states
   chegam em fp16. Envolver `generate` em `torch.autocast("cuda", torch.float16)`.

## ⚠ DEFEITO CONHECIDO no `treino.jsonl` — corrigir antes de reusar

O system prompt declara **4 ferramentas**: `list_dir`, `read_file`,
`run_command`, `write_file`.

Mas as respostas de treino usam **6**, incluindo `git_status` (6 exemplos) e
`append_file` (6 exemplos), **que não existem no schema**.

Isso ensina o modelo a chamar ferramenta inexistente — ou seja, **treina
alucinação nos pesos**, exatamente o que menos se quer num agente autônomo.
Antes de qualquer reuso: ou declara as 6 no schema, ou remove os 12 exemplos.

## Arquivos

- `sanidade.py` — teste de overfit em 1 exemplo, 3 fases (base / adaptador ativo /
  `disable_adapter()`). É o diagnóstico mais rápido se algo quebrar. Já contém
  as duas correções de dtype acima.
- `tokens.py` — checagem de fronteira da máscara de labels, roda em CPU em
  segundos. Provou que o alinhamento estava certo (`prefixo bate? True`).
- `train.py` — treino completo com avaliação antes/depois em prompts fora do
  dataset. Contém a correção do bug de `skip_special_tokens` (ver abaixo).
- `treino.jsonl` — 45 exemplos, com o defeito descrito acima.

## Bug de medição que quase matou o experimento

`train.py` decodificava a saída com `skip_special_tokens=True`. Como
`<tool_call>` é token **especial** adicionado, essa flag **apaga a tag da
string** — a avaliação reportaria `0/8` com o modelo acertando `8/8`.

Está corrigido (`skip_special_tokens=False`), mas é o exemplo canônico do
princípio 9 do `AGENTS.md`: **suspeite da régua antes do modelo.**

## Running it again (needs a 16GB GPU; 4GB local will not train)

```bash
python qwen_tools_lora/train.py
```

The original runs were done on a Colab T4 over an SSH tunnel. That tunnel was
removed from this repository: Colab's terms disallow remote control such as SSH
shells and remote desktops on the free tier, so the script did not belong here.
Use a rented GPU, or a paid Colab plan with a positive compute unit balance.
