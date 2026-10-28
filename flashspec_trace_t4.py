"""Record a step-by-step FlashSpec trace for the web replay.

Run on a Colab Tesla T4 from the FlashSpec repo root, at the 0.3.1 release
commit (c19bac4), after the same setup as notebooks/04_gpu_measurement.ipynb:

    python flashspec_trace_t4.py            # writes flashspec_trace_<commit>.json

It uses FlashSpec's own engine and the benchmark harness's own baseline code,
so every number is comparable to the published 0.3.x results:

* FlashSpec: SpeculativeEngine with drafter_strategy="auto" (composite-g6-f2:
  prompt lookup up to 6 tokens when the context has a match, else the 0.5B
  model drafts 2), compile="auto", per-step profiling on.
* Baseline: benchmarks.harness._hf_generate with a reused StaticCache and
  CompileConfig(), the published "HuggingFace generate compiled" method.
* Both: Qwen2.5-1.5B-Instruct target, Qwen2.5-0.5B-Instruct drafter, fp16,
  greedy, batch 1, chat template, neutral GenerationConfig, max 128 new
  tokens. Unlike the release benchmark, both stop at EOS, so the replay shows
  the answer and not filler after it.

Each prompt runs REPEATS times per method; every wall time is kept, and the
trace kept is the run with the median FlashSpec wall time.

Which prompt the site replays is decided here, before the run: in each
dataset, the prompt whose FlashSpec / HuggingFace speed-up is the median of
that dataset, taking the lower of the two middle prompts when the count is
even (5th of 10 when sorted), never the best one.

Add --smoke to run a tiny model pair on CPU (checks the script, not speed).
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

import flashspec
from benchmarks.harness import _chat_ids, _hf_generate, _neutralise_generation_config
from benchmarks.prompt_sets import load_prompt_set
from flashspec import FlashSpecConfig, SamplingConfig, SpeculativeEngine
from flashspec.engine import HuggingFaceDrafter, TargetModel
from flashspec.engine.ngram import is_lookup_drafter
from flashspec.utils.config import MetricsConfig

DATASETS = ("humaneval_edit", "humaneval")
REPEATS = 3
WARMUP = 2


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def pieces(tokenizer, ids: list[int]) -> list[str]:
    """Text each token adds, from incremental decodes (safe for multi-byte tokens)."""
    out, prev = [], ""
    for i in range(1, len(ids) + 1):
        text = tokenizer.decode(ids[:i], skip_special_tokens=True)
        out.append(text[len(prev):])
        prev = text
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny model pair on CPU, 2 prompts")
    args = ap.parse_args()

    if args.smoke:
        target_id = drafter_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        device, dtype, dtype_name, n_prompts, max_new, repeats, warmup = torch.device("cpu"), torch.float32, "float32", 2, 24, 2, 1
    else:
        target_id, drafter_id = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
        device, dtype, dtype_name, n_prompts, max_new, repeats, warmup = torch.device("cuda"), torch.float16, "float16", 10, 128, REPEATS, WARMUP
        gpu = torch.cuda.get_device_name(0)
        assert "T4" in gpu, f"Run this on a Tesla T4 to match the published numbers (got {gpu})."

    tokenizer = AutoTokenizer.from_pretrained(target_id)
    load = lambda name: AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()  # noqa: E731
    target_module = load(target_id)
    drafter_module = target_module if drafter_id == target_id else load(drafter_id)
    for m in {id(target_module): target_module, id(drafter_module): drafter_module}.values():
        _neutralise_generation_config(m)
    eos = sorted({int(t) for t in ([tokenizer.eos_token_id] + list(getattr(target_module.generation_config, "eos_token_id", None) or [])) if t is not None})

    config = FlashSpecConfig(
        target_name=target_id, drafter_name=drafter_id, device=str(device), dtype=dtype_name,
        max_new_tokens=max_new, eos_token_ids=tuple(eos), drafter_strategy="auto",
        compile="auto", compile_max_cache_len=1024,
        sampling=SamplingConfig(temperature=0.0), metrics=MetricsConfig(profile_steps=True),
    )
    engine = SpeculativeEngine(
        config,
        [HuggingFaceDrafter(drafter_id, device, dtype, model=drafter_module)],
        TargetModel(target_id, device, dtype, model=target_module),
    )
    lookup = [bool(is_lookup_drafter(d)) for d in engine._drafters]  # noqa: SLF001
    static_cache = StaticCache(config=target_module.config, max_cache_len=1024) if device.type == "cuda" else None

    def run_flashspec(prompt):
        sync(device); t0 = time.perf_counter()
        r = engine.generate(prompt, max_new_tokens=max_new)
        sync(device); wall = time.perf_counter() - t0
        n = int(r.n_tokens_generated)
        ids = [int(t) for t in r.output_ids[0, :n].tolist()]
        steps = [
            {"by": "lookup" if lookup[s.drafter] else "model", "proposed": int(s.proposed),
             "accepted": int(s.accepted), "emitted": int(s.emitted), "ms": round(float(s.seconds) * 1e3, 3)}
            for s in r.arm_steps
        ]
        profile = [
            {k: round(float(getattr(p, k)), 3) for k in ("draft_ms", "verify_ms", "sample_ms", "bookkeeping_ms")}
            for p in (r.step_profile or ())
        ]
        return wall, ids, steps, profile, float(r.acceptance_rate)

    def run_hf(prompt):
        sync(device); t0 = time.perf_counter()
        out = _hf_generate(target_module, prompt, max_new, 0.0, do_sample=False, eos_token_ids=eos, static_cache=static_cache)
        sync(device); wall = time.perf_counter() - t0
        return wall, [int(t) for t in out[0].tolist() if int(t) not in eos]

    datasets = {}
    for name in DATASETS:
        records = load_prompt_set(name)[:n_prompts]
        prompts = [torch.tensor([_chat_ids(tokenizer, r["text"])], device=device) for r in records]
        for _ in range(warmup):  # first calls compile and capture CUDA graphs
            run_flashspec(prompts[0]); run_hf(prompts[0])
        rows = []
        for rec, prompt in zip(records, prompts):
            fs_runs = [run_flashspec(prompt) for _ in range(repeats)]
            hf_runs = [run_hf(prompt) for _ in range(repeats)]
            fs_walls = [r[0] for r in fs_runs]; hf_walls = [r[0] for r in hf_runs]
            keep = sorted(range(repeats), key=lambda i: fs_walls[i])[repeats // 2]
            _, ids, steps, profile, acc = fs_runs[keep]
            hf_ids = hf_runs[0][1]
            row = {
                "id": rec["id"], "prompt": rec["text"],
                "flashspec_wall_s": fs_walls, "hf_compiled_wall_s": hf_walls,
                "speedup_median": statistics.median(hf_walls) / statistics.median(fs_walls),
                "tokens": len(ids), "tokens_match_hf": ids == hf_ids,
                "first_mismatch": next((i for i, (a, b) in enumerate(zip(ids, hf_ids)) if a != b), None),
                "acceptance_rate": acc, "steps": steps, "step_profile": profile,
                "pieces": pieces(tokenizer, ids), "hf_tokens": len(hf_ids),
            }
            rows.append(row)
            print(f"{name}/{rec['id']}: {row['speedup_median']:.2f}x  tokens={row['tokens']}  steps={len(steps)}  "
                  f"acceptance={acc:.2f}  match={row['tokens_match_hf']}")
        speedups = [r["speedup_median"] for r in rows]
        chosen = sorted(range(len(rows)), key=lambda i: speedups[i])[(len(rows) - 1) // 2]  # lower median
        datasets[name] = {"selection_rule": "lower median FlashSpec/HF-compiled speed-up in this dataset (5th of 10 sorted)",
                          "chosen": rows[chosen]["id"], "speedups": speedups, "prompts": rows}
        print(f"{name}: speed-ups {', '.join(f'{v:.2f}' for v in speedups)} -> replaying {rows[chosen]['id']}")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "flashspec_commit": git_commit(), "flashspec_version": flashspec.__version__,
        "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "target": target_id, "drafter": drafter_id, "dtype": dtype_name, "greedy": True, "batch": 1,
        "max_new_tokens": max_new, "stop_at_eos": True, "repeats": repeats, "warmup_calls": warmup,
        "strategy": engine.drafter_strategy, "strategy_reason": engine.drafter_strategy_reason,
        "smoke": args.smoke,
    }
    path = f"flashspec_trace_{meta['flashspec_commit']}{'_smoke' if args.smoke else ''}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "datasets": datasets}, fh, ensure_ascii=False, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
