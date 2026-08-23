# Phi-4-mini-instruct-Q4_K_M on NVIDIA GeForce GTX 1650, 2026-08-23

## The machine these numbers came from

| | |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1650 |
| VRAM | 4096 MiB |
| System RAM | 15813 MiB |
| CPU cores | 12 |
| Backend | llama.cpp Vulkan build 10502 (0adcc3bb5) |
| Context served | 8192 (16384 refused: the KV cache would not fit in 4 GiB) |
| Date | 2026-08-23 |

These numbers describe **this machine and this file**, on the date above. They are not a ranking of the models and they do not carry to other hardware, other quantizations of the same weights, or other runtimes. Read a row as: on a machine like this one, this artifact behaved like this. A newer or larger model may well be better and simply not run here.

## The artifact

| | |
| --- | --- |
| File | `Phi-4-mini-instruct-Q4_K_M.gguf` |
| Bytes | 2491874272 |
| SHA-256 | `88c00229914083cd112853aab84ed51b87bdf6b9ce42f532d8c85c7c63b1730a` |
| Served as | `Phi-4-mini-instruct-Q4_K_M` |

The score belongs to this file. It does not belong to the weights it was quantized from, to another quantization of them, or to a fine-tune of them.

## Result

| ruler | result |
| --- | --- |
| HumanEval, 20 problems of 164 | 15/20 = 75.0% pass@1 |
| Generation throughput | 29.37 tok/s median over 20 generations, reported by the server itself |
| Native tool call | **no**, the file did not land as asked (no tool was called) |

The model's own reply to that request, so the reader can see whether the harness failed to offer the tools or the model answered in prose:

```
I will use the write_file tool to create the result.txt file with the specified content.

write_file("result.txt", "alpha\nbeta\n")
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
| HumanEval/0 | pass | 26.77 | 61 |
| HumanEval/8 | pass | 29.71 | 54 |
| HumanEval/16 | pass | 29.73 | 25 |
| HumanEval/24 | pass | 29.83 | 44 |
| HumanEval/32 | fail | 29.47 | 192 |
| HumanEval/41 | fail | 29.54 | 21 |
| HumanEval/49 | pass | 29.59 | 48 |
| HumanEval/57 | pass | 29.55 | 81 |
| HumanEval/65 | fail | 29.49 | 41 |
| HumanEval/73 | pass | 29.38 | 53 |
| HumanEval/82 | pass | 29.47 | 69 |
| HumanEval/90 | pass | 29.35 | 43 |
| HumanEval/98 | pass | 29.24 | 44 |
| HumanEval/106 | pass | 29.2 | 79 |
| HumanEval/114 | pass | 29.27 | 76 |
| HumanEval/122 | fail | 29.11 | 32 |
| HumanEval/131 | fail | 29.28 | 69 |
| HumanEval/139 | pass | 29.3 | 104 |
| HumanEval/147 | pass | 28.94 | 104 |
| HumanEval/155 | pass | 29.06 | 65 |

