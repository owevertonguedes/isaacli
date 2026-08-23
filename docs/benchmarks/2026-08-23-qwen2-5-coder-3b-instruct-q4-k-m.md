# qwen2.5-coder-3b-instruct-q4_k_m on NVIDIA GeForce GTX 1650, 2026-08-23

## The machine these numbers came from

| | |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1650 |
| VRAM | 4096 MiB |
| System RAM | 15813 MiB |
| CPU cores | 12 |
| Backend | llama.cpp Vulkan build 10502 (0adcc3bb5) |
| Context served | 16384 |
| Date | 2026-08-23 |

These numbers describe **this machine and this file**, on the date above. They are not a ranking of the models and they do not carry to other hardware, other quantizations of the same weights, or other runtimes. Read a row as: on a machine like this one, this artifact behaved like this. A newer or larger model may well be better and simply not run here.

## The artifact

| | |
| --- | --- |
| File | `qwen2.5-coder-3b-instruct-q4_k_m.gguf` |
| Bytes | 2104932800 |
| SHA-256 | `724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7` |
| Served as | `qwen2.5-coder-3b-instruct-q4_k_m` |

The score belongs to this file. It does not belong to the weights it was quantized from, to another quantization of them, or to a fine-tune of them.

## Result

| ruler | result |
| --- | --- |
| HumanEval, 20 problems of 164 | 18/20 = 90.0% pass@1 |
| Generation throughput | 36.85 tok/s median over 20 generations, reported by the server itself |
| Native tool call | **no**, the file did not land as asked (no tool was called) |

The model's own reply to that request, so the reader can see whether the harness failed to offer the tools or the model answered in prose:

```
```json
{
  "name": "write_file",
  "arguments": {
    "path": "result.txt",
    "content": "alpha\nbeta"
  }
}
```
```

## How this was measured

- HumanEval was read from `https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz`, whose bytes hash to `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`.
- The 20 problems are a fixed stride over the 164 sorted task ids, so every model is graded on the same ones.
- The judge is the dataset's own `check()`, run inside the project's sandbox (`execution.run_command`), never on the host.
- Temperature 0.
- Throughput is the server's own `predicted_per_second`, so it excludes this client and the socket.
- The tool-call row is judged by the bytes on disk, not by what the model said it did.

## Every problem

| task | verdict | tok/s | generated tokens |
| --- | --- | --- | --- |
| HumanEval/0 | pass | 37.05 | 165 |
| HumanEval/8 | pass | 37.58 | 160 |
| HumanEval/16 | pass | 37.67 | 84 |
| HumanEval/24 | pass | 37.75 | 89 |
| HumanEval/32 | fail | 37.21 | 456 |
| HumanEval/41 | pass | 37.05 | 167 |
| HumanEval/49 | pass | 37.1 | 125 |
| HumanEval/57 | pass | 36.91 | 165 |
| HumanEval/65 | pass | 36.85 | 134 |
| HumanEval/73 | pass | 36.67 | 225 |
| HumanEval/82 | pass | 36.84 | 135 |
| HumanEval/90 | pass | 36.66 | 232 |
| HumanEval/98 | pass | 36.92 | 112 |
| HumanEval/106 | pass | 36.75 | 184 |
| HumanEval/114 | pass | 36.63 | 157 |
| HumanEval/122 | pass | 36.35 | 171 |
| HumanEval/131 | pass | 36.34 | 139 |
| HumanEval/139 | pass | 36.48 | 130 |
| HumanEval/147 | pass | 36.25 | 267 |
| HumanEval/155 | fail | 36.14 | 156 |

