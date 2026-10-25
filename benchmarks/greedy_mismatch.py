"""Diagnose the three greedy mismatches from the Phase 4 T4 run (git 36134c0).

Phase 4 recorded greedy FlashSpec != ``hf_greedy`` on three prompts:

- Qwen2.5-1.5B, MT-Bench prompt index 8, first divergence at token 97
- TinyLlama-1.1B, HumanEval prompt index 7, at token 61
- TinyLlama-1.1B, MT-Bench prompt index 5, at token 0

For each prompt this module reruns both decoders and reports, from
measurements only:

- fp32: whether FlashSpec greedy equals ``hf_greedy`` (it must).
- fp16: the first divergence (if any) and, at that position and at the
  position Phase 4 recorded, the target's top-2 logits and margin as seen
  by FlashSpec (its verify rows) and by ``hf_greedy`` (``output_logits``).
- Token 0: the logits from HuggingFace's full-prompt prefill, from the
  pre-fix path (prefill ``prompt[:-1]``, then ``[last prompt token +
  draft]``), and from FlashSpec as it is now.

A near-tie shows up as a top-2 margin smaller than the difference between
the two computations' logits. A bookkeeping bug shows up as a divergence
in fp32, or a large logit difference at a position where both decoders
saw the same prefix.

Run from the checkout root on a CUDA machine::

    python benchmarks/greedy_mismatch.py --output diagnostics/greedy_mismatch.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.harness import _chat_ids, _load_checkpoint_modules, _load_tokenizer  # noqa: E402
from benchmarks.prompt_sets import load_prompt_set  # noqa: E402
from flashspec.bandit.ucb import UCB1Selector  # noqa: E402
from flashspec.engine.hf_drafter import HuggingFaceDrafter  # noqa: E402
from flashspec.engine.speculative import SpeculativeEngine  # noqa: E402
from flashspec.engine.verifier import TargetModel  # noqa: E402
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig  # noqa: E402

__all__ = ["CASES", "diagnose", "run_case"]

_QWEN = ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct")
_TINY = ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Felladrin/Llama-160M-Chat-v1")

CASES: tuple[dict[str, Any], ...] = (
    {"pair": _QWEN, "dataset": "mt_bench", "index": 8, "phase4_divergence": 97},
    {"pair": _TINY, "dataset": "humaneval", "index": 7, "phase4_divergence": 61},
    {"pair": _TINY, "dataset": "mt_bench", "index": 5, "phase4_divergence": 0},
)
_GAMMA = 4
_NEW_TOKENS = 128
_TOPK = 5


def _top(row: torch.Tensor) -> dict[str, Any]:
    values, ids = row.float().topk(_TOPK)
    return {
        "ids": [int(item) for item in ids],
        "logits": [float(item) for item in values],
        "margin_top1_top2": float(values[0] - values[1]),
    }


def _hf_greedy(target: Any, prompt: torch.Tensor) -> tuple[list[int], list[torch.Tensor]]:  # noqa: ANN401
    with torch.no_grad():
        out = target.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=_NEW_TOKENS,
            min_new_tokens=_NEW_TOKENS,
            eos_token_id=None,
            pad_token_id=target.generation_config.pad_token_id,
            output_logits=True,
            return_dict_in_generate=True,
        )
    tokens = [int(item) for item in out.sequences[0, prompt.shape[1] :].tolist()]
    return tokens, [step[0].float().cpu() for step in out.logits]


def _flashspec_greedy(
    target_module: Any,  # noqa: ANN401
    drafter_module: Any,  # noqa: ANN401
    prompt: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[list[int], dict[int, torch.Tensor]]:
    device = prompt.device
    target = TargetModel("target", device, dtype, model=target_module)
    drafter = HuggingFaceDrafter("drafter", device=device, model=drafter_module)
    prompt_len = int(prompt.shape[1])
    rows_by_position: dict[int, torch.Tensor] = {}
    verify_rows = target._verify_rows

    def _capture(draft: torch.Tensor) -> torch.Tensor:
        kv = target._inc
        committed = kv.cached_length + kv.pending_length
        rows = verify_rows(draft)
        first = committed - prompt_len
        for offset in range(int(rows.shape[1])):
            # A later step that re-covers a position (after a rejection)
            # replaces the earlier row; the emitted token came from the last.
            rows_by_position[first + offset] = rows[0, offset].float().cpu()
        return rows

    target._verify_rows = _capture  # type: ignore[method-assign]
    config = FlashSpecConfig(
        drafter_strategy="bandit",  # the model-drafter path measured before 0.3.0
        device=str(device),
        compile=False,  # eager diagnostic: compared with eager generate()
        dtype=str(dtype).removeprefix("torch."),  # type: ignore[arg-type]
        max_new_tokens=_NEW_TOKENS,
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=_GAMMA, temperature=0.0),
    )
    engine = SpeculativeEngine(
        config,
        [drafter],  # type: ignore[list-item]
        target,
        UCB1Selector(n_arms=1, window_size=0),
    )
    tokens = [int(item) for item in engine.generate(prompt).output_ids[0].tolist()]
    return tokens, rows_by_position


def _token0_paths(target: Any, prompt: torch.Tensor, hf_tokens: list[int]) -> dict[str, Any]:  # noqa: ANN401
    from transformers.cache_utils import DynamicCache

    with torch.no_grad():
        full = target(prompt, use_cache=True, logits_to_keep=1).logits[0, -1].float().cpu()
        cache = DynamicCache()
        target(prompt[:, :-1], past_key_values=cache, use_cache=True, logits_to_keep=1)
        draft = torch.tensor([hf_tokens[:_GAMMA]], dtype=torch.long, device=prompt.device)
        held = (
            target(
                torch.cat([prompt[:, -1:], draft], dim=1),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=_GAMMA + 1,
            )
            .logits[0, 0]
            .float()
            .cpu()
        )
    return {
        "hf_full_prefill": _top(full),
        "prefix_held_back_path_before_fix": _top(held),
        "max_abs_logit_difference": float((full - held).abs().max()),
        "argmax_agrees": int(full.argmax()) == int(held.argmax()),
    }


def run_case(case: dict[str, Any], dtype: torch.dtype, device: torch.device) -> dict[str, Any]:
    """Run one recorded mismatch in one dtype. See the module docstring."""
    target_name, drafter_name = case["pair"]
    config = {"model": target_name, "drafter": drafter_name, "dtype": str(dtype).split(".")[-1]}
    drafter_module, target_module = _load_checkpoint_modules(config, device)
    tokenizer = _load_tokenizer(target_name)
    record = load_prompt_set(case["dataset"])[case["index"]]
    prompt = torch.tensor([_chat_ids(tokenizer, record["text"])], dtype=torch.long, device=device)

    hf_tokens, hf_logits = _hf_greedy(target_module, prompt)
    fs_tokens, fs_rows = _flashspec_greedy(target_module, drafter_module, prompt, dtype)
    limit = min(len(hf_tokens), len(fs_tokens))
    divergence = next((i for i in range(limit) if hf_tokens[i] != fs_tokens[i]), None)

    def _at(position: int) -> dict[str, Any] | None:
        if position >= len(hf_logits) or position not in fs_rows:
            return None
        hf_row, fs_row = hf_logits[position], fs_rows[position]
        hf_top = _top(hf_row)
        return {
            "position": position,
            "same_prefix_before_position": hf_tokens[:position] == fs_tokens[:position],
            "hf_greedy": hf_top,
            "flashspec": _top(fs_row),
            "flashspec_logits_of_hf_top2": [float(fs_row[i]) for i in hf_top["ids"][:2]],
            "max_abs_logit_difference": float((hf_row - fs_row).abs().max()),
        }

    result: dict[str, Any] = {
        "target": target_name,
        "drafter": drafter_name,
        "dataset": case["dataset"],
        "prompt_index": case["index"],
        "prompt_id": record["id"],
        "prompt_tokens": int(prompt.shape[1]),
        "dtype": str(dtype).split(".")[-1],
        "gamma": _GAMMA,
        "matches_hf_greedy": divergence is None and len(hf_tokens) == len(fs_tokens),
        "first_divergence": divergence,
        "phase4_divergence": case["phase4_divergence"],
        "at_first_divergence": _at(divergence) if divergence is not None else None,
        "at_phase4_divergence": _at(int(case["phase4_divergence"])),
        "token0_paths": _token0_paths(target_module, prompt, hf_tokens),
    }
    del drafter_module, target_module
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def diagnose(device: torch.device | None = None) -> dict[str, Any]:
    """Run all three cases in float32 and float16."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtypes = (torch.float32, torch.float16)
    runs = [run_case(case, dtype, device) for case in CASES for dtype in dtypes]
    fp32 = [run for run in runs if run["dtype"] == "float32"]
    return {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cases": runs,
        "fp32_all_match": all(run["matches_hf_greedy"] for run in fp32),
    }


def main(argv: list[str] | None = None) -> int:
    """Write the diagnosis JSON and print a short summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = diagnose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for run in report["cases"]:
        print(
            f"{run['target']} {run['dataset']}[{run['prompt_index']}] {run['dtype']}: "
            f"match={run['matches_hf_greedy']} first_divergence={run['first_divergence']} "
            f"token0 paths agree={run['token0_paths']['argmax_agrees']}"
        )
    return 0 if report["fp32_all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
