# Seed pack Isaac - 2026-07-20

Objetivo: juntar as sementes de comportamento que devem virar treino/LoRA para
subir o nivel do Isaac nos fluxos reais do usuario.

## Datasets incluidos

1. `datasets/task05-andaime-marcadores-gemini-2026-07-20.jsonl`
   - 8 exemplos.
   - Ensina tool-call por marcador: usar `replace_between`, escrever somente
     miolo, chamar `check_file`, nao repetir ferramenta depois de OK.

2. `datasets/commit-assinatura-isaac-2026-07-20.jsonl`
   - 4 exemplos.
   - Ensina commit assinado com trailer Git verificavel e `git log -1 --format=%B`.

3. `datasets/intencao-pergunta-vs-execucao-isaac-2026-07-20.jsonl`
   - 6 exemplos.
   - Ensina distinguir pergunta exploratoria de ordem para executar.

Total atual:

```text
18 exemplos curados
```

## O que ainda falta para "subir o nivel"

Dataset nao muda peso sozinho. O proximo passo real e treinar um LoRA/SFT e
medir antes/depois contra:

- task 05 do andaime;
- commit assinado;
- sessao `2026-07-20-142104-047ad3` ou prompt equivalente de minijogo.

## Politica de adaptador

Nao fundir LoRA no modelo base ainda. Enquanto a melhora nao for comprovada por
antes/depois, cada tentativa deve continuar como adaptador separado. Fusao so
entra quando o adapter virar uma nova geracao mensuravel do Isaac.

## Criterio para aprovar o treino

- melhorar pelo menos um dos tres fluxos sem regredir gravemente os outros;
- nao transformar pergunta em execucao;
- nao declarar sucesso apos comando falho;
- manter custo compativel com o notebook para inferencia.
