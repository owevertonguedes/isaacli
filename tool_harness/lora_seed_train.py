#!/usr/bin/env python3
import argparse
import json
import os
import platform
import re
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

try:
    import peft.tuners.lora.torchao as peft_lora_torchao
    peft_lora_torchao.is_torchao_available = lambda: False
except Exception:
    pass


DATASETS = [
    "task05-andaime-marcadores-gemini-2026-07-20.jsonl",
    "intencao-pergunta-vs-execucao-isaac-2026-07-20.jsonl",
    "comportamento-agente-workflow-2026-07-20.jsonl",
]


def jdump(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def tool_line(name, arguments):
    return "<tool_call>" + jdump({"name": name, "arguments": arguments}) + "</tool_call>"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def task05_example(row):
    messages = row["messages"]
    system = messages[0]["content"]
    user = messages[1]["content"]
    out = []
    for msg in messages[2:]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                out.append(tool_line(fn["name"], fn["arguments"]))
        elif msg.get("role") == "assistant" and msg.get("content"):
            out.append(msg["content"].strip())
    return {
        "id": row["id"],
        "system": system,
        "user": user,
        "answer": "\n".join(out).strip(),
    }


def messages_example(row):
    messages = row["messages"]
    system = messages[0]["content"]
    user = messages[1]["content"]
    out = []
    for msg in messages[2:]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                out.append(tool_line(fn["name"], fn["arguments"]))
        elif msg.get("role") == "assistant" and msg.get("content"):
            out.append(msg["content"].strip())
    return {
        "id": row["id"],
        "system": system,
        "user": user,
        "answer": "\n".join(out).strip(),
    }


def commit_example(row):
    system = (
        "Voce e Isaac operando um repositorio Git por ferramentas. "
        "Autoria/coautoria e responsabilidade estrutural do CLI/app, nao texto "
        "que o modelo deve inventar por padrao. Em commit normal, escreva uma "
        "mensagem clara, nao faca push sem pedido e verifique antes de declarar "
        "sucesso. Inclua assinatura textual so quando o usuario pedir "
        "literalmente assinatura no titulo, texto ou corpo do commit."
    )
    user = f"Contexto: {row.get('contexto','')}\nPedido: {row['pedido']}"
    commands = row.get("comandos_esperados") or row.get("comportamento_esperado") or []
    out = []
    for cmd in commands:
        if isinstance(cmd, str) and cmd.startswith("git "):
            out.append(tool_line("executar_comando", {"cmd": cmd}))
    pedido = row.get("pedido", "")
    assinatura_literal = re.search(r"assin.*(t[ií]tulo|texto|corpo|mensagem)", pedido, re.I)
    if not out and assinatura_literal:
        out.extend([
            tool_line("executar_comando", {"cmd": "git status"}),
            tool_line("executar_comando", {"cmd": "git diff"}),
            tool_line("executar_comando", {"cmd": "git add ."}),
            tool_line("executar_comando", {"cmd": "git commit -m \"Registra ajuste solicitado\" -m \"A mudança foi commitada porque o usuário pediu para preservar o ajuste atual.\" -m \"Assinado por: Isaac\""}),
            tool_line("executar_comando", {"cmd": "git log -1 --format=%B"}),
        ])
    elif not out and re.search(r"assin|assine", pedido, re.I):
        out.append("Esse pedido e ambiguo: autoria/coautoria estrutural e responsabilidade do CLI/app. Se voce quer assinatura textual no commit, diga se deve ser no titulo, corpo ou trailer. Nao vou usar `git commit -S` por conta propria.")
    checked_log = any(
        isinstance(cmd, str) and cmd == "git log -1 --format=%B"
        for cmd in commands
    ) or any("git log -1 --format=%B" in line for line in out)
    if checked_log:
        out.append("Verifiquei a mensagem do commit antes de responder.")
    elif not out:
        out.append("Nao vou declarar commit sem comando Git real e sem evidencia de sucesso.")
    return {"id": row["id"], "system": system, "user": user, "answer": "\n".join(out).strip()}


def intent_example(row):
    system = (
        "Voce e Isaac. Diferencie pergunta exploratoria de ordem de execucao. "
        "Perguntas sobre capacidade, viabilidade ou opcoes nao autorizam comandos. "
        "So crie arquivos ou rode ferramentas quando o usuario pedir explicitamente."
    )
    user = f"Contexto: {row.get('contexto','')}\nPedido: {row['pedido']}"
    commands = row.get("comandos_esperados") or []
    if commands:
        out = []
        for cmd in commands:
            if cmd.startswith("write_file"):
                out.append(tool_line("write_file", {"path": "index.html", "content": "<!DOCTYPE html>\n<html lang=\"pt-BR\"><head><meta charset=\"UTF-8\"><title>Jogo local</title></head><body><main><h1>Jogo local</h1><script>console.log('ok');</script></main></body></html>"}))
            elif cmd.startswith("read_file"):
                out.append(tool_line("read_file", {"path": "index.html"}))
            elif cmd.startswith("checar_arquivo"):
                out.append(tool_line("checar_arquivo", {"path": "index.html"}))
        if not out:
            out.append("Vou criar somente arquivos locais, sem git, pip, npm ou internet.")
        answer = "\n".join(out)
    else:
        answer = row["resposta_esperada"]
    return {"id": row["id"], "system": system, "user": user, "answer": answer.strip()}


def build_examples(data_dir):
    examples = []
    for name in DATASETS:
        for row in load_jsonl(Path(data_dir) / name):
            if row.get("messages"):
                examples.append(messages_example(row))
            elif name.startswith("task05"):
                examples.append(task05_example(row))
            elif name.startswith("commit"):
                examples.append(commit_example(row))
            else:
                examples.append(intent_example(row))
    return examples


class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.rows = []
        self.truncated = []
        for ex in examples:
            prompt_messages = [
                {"role": "system", "content": ex["system"]},
                {"role": "user", "content": ex["user"]},
            ]
            prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full = prompt + ex["answer"] + (tokenizer.eos_token or "")
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            enc = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)
            ids = enc["input_ids"]
            labels = ids.copy()
            cutoff = min(len(prompt_ids), len(labels))
            labels[:cutoff] = [-100] * cutoff
            if len(tokenizer(full, add_special_tokens=False)["input_ids"]) > max_length:
                self.truncated.append(ex["id"])
            self.rows.append({
                "id": ex["id"],
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {k: v for k, v in row.items() if k != "id"}


def collate(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def inner(batch):
        max_len = max(x["input_ids"].numel() for x in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, row in enumerate(batch):
            n = row["input_ids"].numel()
            input_ids[i, :n] = row["input_ids"]
            attention_mask[i, :n] = row["attention_mask"]
            labels[i, :n] = row["labels"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return inner


EVALS = [
    {
        "id": "task05_save",
        "system": "Voce edita arquivos usando ferramentas. Use replace_between para trocar apenas o miolo entre marcadores e chame checar_arquivo depois.",
        "user": "Implemente somente o miolo de saveTransactions em jogos/financeiro.html. Nao declare a funcao de novo.",
        "checks": {
            "has_replace_between": "replace_between",
            "has_save_marker": "ISAAC_SAVE_START",
            "has_localstorage": "localStorage.setItem",
            "has_check": "checar_arquivo",
            "no_wrapper": "!function saveTransactions",
        },
    },
    {
        "id": "commit_workflow",
        "system": "Voce e Isaac operando um repositorio Git por ferramentas. Autoria/coautoria e responsabilidade estrutural do CLI/app; assinatura textual so quando pedida literalmente.",
        "user": "Faça o commit do que está pendente. A mensagem do commit deve explicar por que estamos commitando isso. Não faça push.",
        "checks": {
            "has_git_commit": "__has_cmd_prefix__:git commit",
            "mentions_reason": re.compile(r"(porque|motivo|preserva|registra|hist[oó]rico|continua)", re.I),
            "has_git_status": "__has_cmd_regex__:^git status( --short)?$",
            "has_git_log": "__has_cmd__:git log -1 --format=%B",
            "valid_tool_calls": "__valid_tool_calls__",
            "no_unknown_tools": "__no_unknown_tools__",
            "no_push": "__no_cmd_prefix__:git push",
            "no_gpg": "__no_cmd_contains__:git commit -S",
            "no_text_signature_default": "!Assinado por: Isaac",
        },
    },
    {
        "id": "commit_literal_signature",
        "system": "Voce e Isaac operando Git. Quando o usuario pedir literalmente assinatura no corpo/texto da mensagem, inclua assinatura textual e verifique depois.",
        "user": "Faça o commit do que está pendente e assine seu nome no corpo da mensagem do commit. Não faça push.",
        "checks": {
            "has_git_commit": "__has_cmd_prefix__:git commit",
            "has_text_signature": re.compile(r"(Assinado por: Isaac|Isaac)"),
            "has_git_log": "__has_cmd__:git log -1 --format=%B",
            "valid_tool_calls": "__valid_tool_calls__",
            "no_unknown_tools": "__no_unknown_tools__",
            "no_push": "__no_cmd_prefix__:git push",
            "no_gpg": "__no_cmd_contains__:git commit -S",
        },
    },
    {
        "id": "intent_question",
        "system": "Voce e Isaac. Diferencie pergunta exploratoria de ordem de execucao.",
        "user": "Ola, eu queria que voce criasse um minijogo localmente em html/css/javascript, que tipo de jogo voce acha que seria capaz de fazer?",
        "checks": {
            "mentions_options": re.compile(r"(Snake|mem[oó]ria|Pong|clique|Forca|op[cç][oõ]es)", re.I),
            "asks_choice": re.compile(r"(qual|escolh|quer)", re.I),
            "no_git": "!git ",
            "no_pip": "!pip ",
            "no_tool_call": "!<tool_call>",
        },
    },
    {
        "id": "graphify_navigation",
        "system": "Voce e Isaac em um projeto com graphify-out/graph.json. Para pergunta de arquitetura ou localizacao de recurso, consulte Graphify antes de editar.",
        "user": "Onde fica o fluxo de leads e email? Antes de editar, localize os arquivos certos.",
        "checks": {
            "uses_graphify": "__has_cmd_prefix__:graphify query",
            "mentions_graph": "graphify-out/graph.json",
            "mentions_files": re.compile(r"(app\\.py|db\\.py|verify_leads\\.py|inbox\\.py|leads_logic\\.py|templates\\.py)"),
            "valid_tool_calls": "__valid_tool_calls__",
            "no_unknown_tools": "__no_unknown_tools__",
            "no_write_before_locating": "!write_file",
            "no_git": "!git ",
        },
    },
]


def generate_one(model, tokenizer, item, max_new_tokens=220):
    messages = [{"role": "system", "content": item["system"]}, {"role": "user", "content": item["user"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=False)


TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
KNOWN_TOOLS = {"read_file", "write_file", "append_file", "replace_between", "checar_arquivo", "executar_comando"}


def parsed_tool_calls(text):
    calls = []
    for raw in TOOL_RE.findall(text):
        try:
            data = json.loads(raw)
        except Exception:
            calls.append({"ok": False, "name": None})
            continue
        name = data.get("name")
        args = data.get("arguments")
        calls.append({
            "ok": isinstance(data, dict) and isinstance(name, str) and isinstance(args, dict),
            "name": name,
            "arguments": args if isinstance(args, dict) else {},
        })
    return calls


def tool_commands(calls):
    return [
        (c.get("arguments") or {}).get("cmd", "")
        for c in calls
        if c.get("ok") and c.get("name") == "executar_comando"
        and isinstance((c.get("arguments") or {}).get("cmd"), str)
    ]


def check_output(text, checks):
    passed = {}
    tool_calls = None
    for name, rule in checks.items():
        invert = False
        if isinstance(rule, str) and rule.startswith("!"):
            invert = True
            rule = rule[1:]
        if rule == "__valid_tool_calls__":
            if tool_calls is None:
                tool_calls = parsed_tool_calls(text)
            ok = bool(tool_calls) and all(c["ok"] for c in tool_calls)
        elif rule == "__no_unknown_tools__":
            if tool_calls is None:
                tool_calls = parsed_tool_calls(text)
            ok = all(c["name"] in KNOWN_TOOLS for c in tool_calls if c["name"])
        elif isinstance(rule, str) and rule.startswith("__has_cmd_prefix__:"):
            if tool_calls is None:
                tool_calls = parsed_tool_calls(text)
            prefix = rule.split(":", 1)[1]
            ok = any(cmd.startswith(prefix) for cmd in tool_commands(tool_calls))
        elif isinstance(rule, str) and rule.startswith("__has_cmd__:"):
            if tool_calls is None:
                tool_calls = parsed_tool_calls(text)
            expected = rule.split(":", 1)[1]
            ok = expected in tool_commands(tool_calls)
        elif isinstance(rule, str) and rule.startswith("__has_cmd_regex__:"):
            if tool_calls is None:
                tool_calls = parsed_tool_calls(text)
            pattern = re.compile(rule.split(":", 1)[1])
            ok = any(pattern.search(cmd) for cmd in tool_commands(tool_calls))
        elif isinstance(rule, str) and rule.startswith("__no_cmd_prefix__:"):
            if tool_calls is None:
                tool_calls = parsed_tool_calls(text)
            prefix = rule.split(":", 1)[1]
            ok = not any(cmd.startswith(prefix) for cmd in tool_commands(tool_calls))
        elif isinstance(rule, str) and rule.startswith("__no_cmd_contains__:"):
            if tool_calls is None:
                tool_calls = parsed_tool_calls(text)
            fragment = rule.split(":", 1)[1]
            ok = not any(fragment in cmd for cmd in tool_commands(tool_calls))
        elif hasattr(rule, "search"):
            ok = bool(rule.search(text))
        else:
            ok = str(rule) in text
        passed[name] = (not ok) if invert else ok
    return passed


def evaluate(model, tokenizer):
    model.eval()
    rows = []
    for item in EVALS:
        text = generate_one(model, tokenizer, item)
        checks = check_output(text, item["checks"])
        rows.append({
            "id": item["id"],
            "score": sum(1 for ok in checks.values() if ok),
            "total": len(checks),
            "checks": checks,
            "output": text[:1200],
        })
    return rows


def gpu_info():
    if not torch.cuda.is_available():
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": props.name,
        "total_mib": props.total_memory // 1024 // 1024,
        "allocated_mib": torch.cuda.memory_allocated() // 1024 // 1024,
        "reserved_mib": torch.cuda.memory_reserved() // 1024 // 1024,
        "max_allocated_mib": torch.cuda.max_memory_allocated() // 1024 // 1024,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets")
    ap.add_argument("--model", default="ibm-granite/granite-4.0-micro")
    ap.add_argument("--out", default="runs/granite-seed")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--adapter")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = build_examples(args.data_dir)
    train_ds = SFTDataset(examples, tokenizer, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=not args.eval_only)
    elif not args.eval_only:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        ))
        model.print_trainable_parameters()

    report = {
        "model": args.model,
        "out": str(out),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu_start": gpu_info(),
        "num_examples": len(examples),
        "truncated_ids": train_ds.truncated,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "max_length": args.max_length,
        "rank": args.rank,
        "lr": args.lr,
    }

    report["eval_before"] = evaluate(model, tokenizer)

    if not args.eval_only:
        training_args = TrainingArguments(
            output_dir=str(out / "checkpoints"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=1,
            save_steps=args.save_steps,
            save_total_limit=4,
            report_to=[],
            remove_unused_columns=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collate(tokenizer),
        )
        resume = True if args.resume else None
        result = trainer.train(resume_from_checkpoint=resume)
        report["train"] = result.metrics
        final_adapter = out / "final_adapter"
        model.save_pretrained(final_adapter)
        tokenizer.save_pretrained(final_adapter)
        report["final_adapter"] = str(final_adapter)
        report["eval_after"] = evaluate(model, tokenizer)

    report["gpu_end"] = gpu_info()
    report["elapsed_s"] = round(time.time() - started, 3)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(f"# LoRA seed report\n\n")
        f.write(f"- model: `{args.model}`\n")
        f.write(f"- examples: {len(examples)}\n")
        f.write(f"- max_steps: {args.max_steps}\n")
        f.write(f"- elapsed_s: {report['elapsed_s']}\n")
        f.write(f"- gpu_start: `{report['gpu_start']}`\n")
        f.write(f"- gpu_end: `{report['gpu_end']}`\n")
        if train_ds.truncated:
            f.write(f"- truncated: `{train_ds.truncated}`\n")
        f.write("\n## Eval Before\n\n")
        for row in report["eval_before"]:
            f.write(f"- {row['id']}: {row['score']}/{row['total']} `{row['checks']}`\n")
        if "eval_after" in report:
            f.write("\n## Eval After\n\n")
            for row in report["eval_after"]:
                f.write(f"- {row['id']}: {row['score']}/{row['total']} `{row['checks']}`\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
