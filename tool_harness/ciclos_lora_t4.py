#!/usr/bin/env python3
"""Orquestra um ciclo LoRA no Colab/T4 a partir da Oficina.

Isto e instrumentacao de treino, nao solucao de produto: empacota o seed pack,
envia para o Colab atual, roda o trainer com checkpoints, copia adapter/relatorio
de volta e atualiza `reports/lora/`.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
REMOTO_BASE = "~/llm-local-ciclos-lora"
DATASETS = [
    "task05-andaime-marcadores-gemini-2026-07-20.jsonl",
    "intencao-pergunta-vs-execucao-isaac-2026-07-20.jsonl",
    "comportamento-agente-workflow-2026-07-20.jsonl",
]


def log(msg):
    print(msg, flush=True)


def limpar_host(host):
    h = (host or "").strip()
    h = re.sub(r"^https?://", "", h)
    h = h.strip().strip("/")
    if not h:
        raise SystemExit("ERRO: informe o CF_HOST do Colab.")
    if any(c.isspace() for c in h):
        raise SystemExit("ERRO: CF_HOST nao pode conter espacos.")
    if not h.endswith(".trycloudflare.com"):
        log("AVISO: CF_HOST nao termina com .trycloudflare.com; vou tentar mesmo assim.")
    return h


def run(cmd, *, cwd=RAIZ, input_file=None, output_file=None, timeout=None):
    stdin = input_file.open("rb") if input_file else None
    stdout = output_file.open("wb") if output_file else None
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            stdin=stdin,
            stdout=stdout or subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            timeout=timeout,
            check=False,
        )
    finally:
        if stdin:
            stdin.close()
        if stdout:
            stdout.close()


def run_text(cmd, *, cwd=RAIZ, timeout=None):
    r = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    if r.stdout:
        print(r.stdout, end="" if r.stdout.endswith("\n") else "\n", flush=True)
    return r


def conectar(host, comando):
    """Build the remote-execution command for the training host.

    This used to shell out to a Colab SSH tunnel, which was removed: Colab's
    terms disallow remote control such as SSH shells on the free tier. Point
    REMOTE_RUNNER at your own runner (a rented GPU box, a paid Colab plan) that
    accepts `<runner> <host> <command>` and reads stdin.
    """
    runner = os.environ.get("REMOTE_RUNNER")
    if not runner:
        raise SystemExit(
            "REMOTE_RUNNER is not set. This script needs a remote GPU runner; "
            "the bundled Colab tunnel was removed for terms-of-service reasons.")
    return [runner, host, comando]


def preparar_pacote():
    tmp = Path(tempfile.mkdtemp(prefix="lora-ciclo-pack-"))
    (tmp / "datasets").mkdir()
    shutil.copy2(AQUI / "lora_seed_train.py", tmp / "lora_seed_train.py")
    for nome in DATASETS:
        shutil.copy2(RAIZ / "datasets" / nome, tmp / "datasets" / nome)
    pacote = tmp.with_suffix(".tgz")
    with tarfile.open(pacote, "w:gz") as tar:
        for item in tmp.rglob("*"):
            tar.add(item, arcname=item.relative_to(tmp))
    return tmp, pacote


def extrair_relatorios(arquivo, run_name):
    destino = RAIZ / "reports" / "lora" / run_name
    destino.mkdir(parents=True, exist_ok=True)
    with tarfile.open(arquivo, "r:gz") as tar:
        for membro in tar.getmembers():
            if membro.name in {"report.json", "report.md"}:
                tar.extract(membro, destino)
    return destino


def atualizar_indice():
    base = RAIZ / "reports" / "lora"
    runs = []
    for report in sorted(base.glob("*/report.json")):
        try:
            r = json.loads(report.read_text())
        except Exception:
            continue
        run = report.parent.name
        runs.append({
            "run": run,
            "model": r.get("model"),
            "examples": r.get("num_examples"),
            "max_steps": r.get("max_steps"),
            "save_steps": r.get("save_steps"),
            "train": r.get("train"),
            "eval_before": [
                {"id": x["id"], "score": x["score"], "total": x["total"]}
                for x in r.get("eval_before", [])
            ],
            "eval_after": [
                {"id": x["id"], "score": x["score"], "total": x["total"]}
                for x in r.get("eval_after", [])
            ],
            "report": str(report.relative_to(RAIZ)),
            "adapter_archive": str((RAIZ / "lora_runs" / "oficina" / f"{run}-lite.tgz").relative_to(RAIZ)),
        })
    out = base / "index.json"
    out.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=2))
    return out


def resumo_report(report_path):
    r = json.loads(report_path.read_text())
    log("")
    log("=== RESUMO DO CICLO ===")
    log(f"modelo: {r.get('model')}")
    log(f"steps: {r.get('max_steps')}  save_steps: {r.get('save_steps')}")
    train = r.get("train") or {}
    if train:
        log(f"tempo_treino: {train.get('train_runtime')}s  step/s: {train.get('train_steps_per_second')}")
        log(f"loss: {train.get('train_loss')}")
    antes = {x["id"]: f"{x['score']}/{x['total']}" for x in r.get("eval_before", [])}
    depois = {x["id"]: f"{x['score']}/{x['total']}" for x in r.get("eval_after", [])}
    for chave in sorted(set(antes) | set(depois)):
        log(f"{chave}: {antes.get(chave, '?')} -> {depois.get(chave, '?')}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-host", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--model", default="ibm-granite/granite-4.0-micro")
    ap.add_argument("--run-name")
    args = ap.parse_args(argv)

    host = limpar_host(args.cf_host)
    run_name = args.run_name or f"oficina-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-s{args.steps}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise SystemExit("ERRO: run-name deve conter so letras, numeros, ponto, underscore ou hifen.")

    log(f"CF_HOST={host}")
    log("checando T4...")
    r = run_text(conectar(host, "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader"), timeout=60)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

    tmp, pacote = preparar_pacote()
    remoto = f"{REMOTO_BASE}/{run_name}"
    try:
        log(f"enviando pacote para {remoto}...")
        cmd_prep = f"rm -rf {remoto} && mkdir -p {remoto} && tar -xzf - -C {remoto}"
        r = run(conectar(host, cmd_prep), input_file=pacote, timeout=180)
        if r.returncode != 0:
            sys.stdout.buffer.write(r.stdout or b"")
            raise SystemExit(r.returncode)

        log("iniciando treino remoto...")
        remote_cmd = (
            f"cd {remoto} && python3 lora_seed_train.py "
            f"--model {args.model} --out runs/{run_name} "
            f"--max-steps {args.steps} --save-steps {args.save_steps} "
            f"--max-length {args.max_length} --rank {args.rank} --lr {args.lr}"
        )
        r = run_text(conectar(host, remote_cmd), timeout=None)
        if r.returncode != 0:
            raise SystemExit(r.returncode)

        destino = RAIZ / "lora_runs" / "oficina"
        destino.mkdir(parents=True, exist_ok=True)
        arquivo = destino / f"{run_name}-lite.tgz"
        log(f"copiando adapter/relatorio para {arquivo}...")
        copy_cmd = (
            f"cd {remoto}/runs/{run_name} && "
            "last=$(ls -d checkpoints/checkpoint-* 2>/dev/null | sort -V | tail -1); "
            "if [ -n \"$last\" ]; then "
            "tar -czf - report.json report.md final_adapter "
            "\"$last/adapter_model.safetensors\" \"$last/adapter_config.json\" \"$last/trainer_state.json\"; "
            "else tar -czf - report.json report.md final_adapter; fi"
        )
        r = run(conectar(host, copy_cmd), output_file=arquivo, timeout=600)
        if r.returncode != 0:
            sys.stdout.buffer.write(r.stdout or b"")
            raise SystemExit(r.returncode)

        rel_dir = extrair_relatorios(arquivo, run_name)
        indice = atualizar_indice()
        log(f"relatorio: {rel_dir / 'report.json'}")
        log(f"indice: {indice}")
        resumo_report(rel_dir / "report.json")
        log("ciclo concluido. Adapter separado; nada foi fundido no modelo base.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            pacote.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
