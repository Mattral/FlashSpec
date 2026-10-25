"""Width-7 greedy mismatches in bundle 3e38251: a padding bug, or fp16 rounding.

In the Phase 6 T4 bundle every greedy mismatch against ``hf_greedy_compiled``
was in the verify-width-7 group (qwen-g6, ngram-g6, both bandits); width 3
and 5 matched everywhere. This module separates the two explanations, from
measurements only:

- **fp32 token-exactness.** Greedy FlashSpec with the target's verify padded
  to width 7, padded to width 5, and unpadded, against HuggingFace eager
  and HuggingFace on a static cache. In fp32 every path must give the same
  tokens; a padding bug would show up here.
- **Pad-content independence.** For a fixed prefix, the logit row of the
  first position of a width-7 call is computed with several different
  tokens in the other six slots. It must be bit-identical: causal
  attention, per-row norms and per-row matmuls never mix rows, and the
  cache entries written by the other slots are past the row's position.
- **Margins at each divergence (fp16).** At the first token where a path
  departs from the reference, the reference's and the path's top-2 logits
  and margins, and the largest logit difference between the two rows. A
  rounding flip shows up as a top-2 margin no larger than that difference,
  on a prefix both paths shared.

The cases are the divergences recorded in bundle ``3e38251`` (Qwen2.5-1.5B
target). On CPU, where Qwen2.5-1.5B in fp32 does not fit in memory, the
same prompts run with another target (``--target``), as a token-exactness
check only.

    python benchmarks/width_diagnostic.py --output diagnostics/width.json   # CUDA
    python benchmarks/width_diagnostic.py --output /tmp/w.json \
        --target Qwen/Qwen2.5-0.5B-Instruct --dtypes float32               # CPU
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.harness import _chat_ids, _load_tokenizer  # noqa: E402
from benchmarks.prompt_sets import load_prompt_set  # noqa: E402
from flashspec.bandit.ucb import UCB1Selector  # noqa: E402
from flashspec.engine.arms import ArmSpec  # noqa: E402
from flashspec.engine.ngram import NgramDrafter  # noqa: E402
from flashspec.engine.speculative import SpeculativeEngine  # noqa: E402
from flashspec.engine.static_cache import StaticKvSession  # noqa: E402
from flashspec.engine.verifier import TargetModel  # noqa: E402
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig  # noqa: E402

__all__ = ["CASES", "diagnose", "pad_independence", "run_case"]

_TARGET = "Qwen/Qwen2.5-1.5B-Instruct"

#: (dataset, prompt id, first divergence vs hf_greedy_compiled, which methods) in bundle 3e38251.
CASES: tuple[dict[str, Any], ...] = (
    {"dataset": "mt_bench", "id": "81", "bundle_divergence": 74, "group": "verify_width_7"},
    {"dataset": "mt_bench", "id": "101", "bundle_divergence": 44, "group": "verify_width_7"},
    {
        "dataset": "humaneval",
        "id": "HumanEval/53",
        "bundle_divergence": 94,
        "group": "verify_width_7",
    },
    {
        "dataset": "mt_bench",
        "id": "82",
        "bundle_divergence": 97,
        "group": "legacy (unpadded) bandit",
    },
)
_NEW_TOKENS = 128
_TOPK = 5
#: FlashSpec paths: (name, lookup gamma, padded verify width or None for unpadded).
#: gamma 6 fills width 7 with real tokens after its first step (as qwen-g6 and
#: ngram-g6 did in the bundle); gamma 2 at width 7 pads 4 of 7 slots on every
#: step (as the bandit's gamma-2 arms did).
_PATHS: tuple[tuple[str, int, int | None], ...] = (
    ("flashspec_g6_padded_w7", 6, 7),
    ("flashspec_g6_unpadded", 6, None),
    ("flashspec_g4_padded_w5", 4, 5),
    ("flashspec_g2_padded_w7", 2, 7),
    ("flashspec_g2_unpadded", 2, None),
)


def _top(row: torch.Tensor) -> dict[str, Any]:
    values, ids = row.float().topk(_TOPK)
    return {
        "ids": [int(item) for item in ids],
        "logits": [float(item) for item in values],
        "margin_top1_top2": float(values[0] - values[1]),
    }


def _hf_paths(
    target: Any,  # noqa: ANN401
    prompt: torch.Tensor,
    static_cache: Any,  # noqa: ANN401
) -> dict[str, tuple[list[int], list[torch.Tensor]]]:
    """HuggingFace eager and static-cache greedy decodes, with per-step logits."""
    from transformers.generation.configuration_utils import CompileConfig

    out: dict[str, tuple[list[int], list[torch.Tensor]]] = {}
    # hf_static is HuggingFace generate on a StaticCache with CompileConfig():
    # compiled on CUDA (the bundle's hf_greedy_compiled), eager on CPU.
    for name, static in (("hf_eager", False), ("hf_static", True)):
        kwargs: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": _NEW_TOKENS,
            "min_new_tokens": _NEW_TOKENS,
            "eos_token_id": None,
            "pad_token_id": target.generation_config.pad_token_id,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "output_logits": True,
            "return_dict_in_generate": True,
        }
        if static:
            static_cache.reset()
            kwargs.update({"past_key_values": static_cache, "compile_config": CompileConfig()})
        with torch.no_grad():
            result = target.generate(prompt, attention_mask=torch.ones_like(prompt), **kwargs)
        tokens = [int(item) for item in result.sequences[0, prompt.shape[1] :].tolist()]
        out[name] = (tokens, [step[0].float().cpu() for step in result.logits])
    return out


def _path_target(target_module: Any, width: int | None, cache_len: int) -> TargetModel:  # noqa: ANN401
    """Build a target wrapper on its own static session (compiled on CUDA), reused per prompt."""
    device = next(target_module.parameters()).device
    dtype = next(target_module.parameters()).dtype
    target = TargetModel("target", device, dtype, model=target_module)
    target.use_static_cache(
        cache_len,
        compile_forward=device.type == "cuda",
        pad_widths=(width,) if width is not None else None,
    )
    return target


def _flashspec_path(
    target: TargetModel,
    prompt: torch.Tensor,
    gamma: int,
) -> tuple[list[int], dict[int, torch.Tensor], list[int]]:
    """Greedy FlashSpec with the n-gram arm: tokens, verify rows by position, call widths."""
    device = prompt.device
    dtype = target._dtype
    session_before = dict(getattr(target._inc, "graph_call_shapes", {}))
    prompt_len = int(prompt.shape[1])
    rows_by_position: dict[int, torch.Tensor] = {}
    verify_rows = target._verify_rows

    def _capture(draft: torch.Tensor) -> torch.Tensor:
        session = target._inc
        committed = session.cached_length + session.pending_length
        rows = verify_rows(draft)
        first = committed - prompt_len
        for offset in range(int(rows.shape[1])):
            rows_by_position[first + offset] = rows[0, offset].float().cpu()
        return rows

    target._verify_rows = _capture  # type: ignore[method-assign]
    logging.getLogger("flashspec.engine.speculative").setLevel(logging.ERROR)
    config = FlashSpecConfig(
        device=str(device),
        compile=False,  # the session above was set up by hand
        dtype=str(dtype).removeprefix("torch."),  # type: ignore[arg-type]
        max_new_tokens=_NEW_TOKENS,
        bandit=BanditConfig(n_arms=1, window_size=0),
        sampling=SamplingConfig(gamma=gamma, temperature=0.0),
    )
    engine = SpeculativeEngine(
        config,
        [NgramDrafter()],  # type: ignore[list-item]
        target,
        UCB1Selector(n_arms=1, window_size=0),
        arms=[ArmSpec(0, gamma)],
    )
    try:
        tokens = [int(item) for item in engine.generate(prompt).output_ids[0].tolist()]
    finally:
        del target._verify_rows  # back to the class method
    session = target._inc
    widths = (
        sorted(
            {
                key[0]
                for key, count in session.graph_call_shapes.items()
                if count > session_before.get(key, 0)
            }
        )
        if isinstance(session, StaticKvSession)
        else []
    )
    return tokens, rows_by_position, widths


def pad_independence(
    target_module: Any,  # noqa: ANN401
    prefix: torch.Tensor,
    width: int = 7,
    trials: int = 4,
    session: StaticKvSession | None = None,
) -> dict[str, Any]:
    """Row 0 of a width-``width`` call under different tokens in the other slots.

    ``prefix`` is forwarded except its last token; each trial then forwards
    ``[last token, other tokens...]`` (width tokens, no padding needed) and
    crops them again. Bit-identical rows mean no other slot reaches row 0.
    """
    device = prefix.device
    if session is None:
        session = StaticKvSession(
            target_module,
            int(prefix.shape[1]) + 2 * width + 8,
            compile_forward=device.type == "cuda",
            pad_widths=(width,),
        )
    session.reserve(int(prefix.shape[1]) + 2 * width + 8)
    session.start(prefix, hold_last=True)
    last = prefix[:, -1:]
    vocab = int(target_module.config.vocab_size)
    generator = torch.Generator().manual_seed(0)
    rows: list[torch.Tensor] = []
    for trial in range(trials):
        others = torch.randint(0, vocab, (1, width - 1), generator=generator).to(device)
        if trial == 0:
            others = last.expand(1, width - 1)  # what the engine's own padding uses
        session._pending = last  # noqa: SLF001
        logits = session.advance(others, keep=width)
        rows.append(logits[0, 0].float().cpu())
        session.crop(width)
    identical = all(torch.equal(rows[0], row) for row in rows[1:])
    return {
        "width": width,
        "trials": trials,
        "row0_bit_identical": identical,
        "max_abs_difference": max(float((rows[0] - row).abs().max()) for row in rows[1:]),
    }


def _divergence(a: list[int], b: list[int]) -> int | None:
    limit = min(len(a), len(b))
    first = next((i for i in range(limit) if a[i] != b[i]), None)
    if first is None and len(a) != len(b):
        return limit
    return first


def run_case(
    case: dict[str, Any],
    target_module: Any,  # noqa: ANN401
    tokenizer: Any,  # noqa: ANN401
    device: torch.device,
    *,
    targets: dict[str, TargetModel],
    hf_cache: Any,  # noqa: ANN401
    pad_session: StaticKvSession,
) -> dict[str, Any]:
    """All paths on one prompt in the module's dtype."""
    prompt = _prompt(case, tokenizer, device)
    hf = _hf_paths(target_module, prompt, hf_cache)
    reference_tokens, reference_logits = hf["hf_static"]
    paths: dict[str, Any] = {}
    for name, (tokens, _logits) in hf.items():
        paths[name] = {"divergence_vs_hf_static": _divergence(tokens, reference_tokens)}
    for name, gamma, _width in _PATHS:
        tokens, rows, widths = _flashspec_path(targets[name], prompt, gamma)
        first = _divergence(tokens, reference_tokens)
        entry: dict[str, Any] = {
            "verify_call_widths": widths,
            "divergence_vs_hf_static": first,
            "divergence_vs_hf_eager": _divergence(tokens, hf["hf_eager"][0]),
        }
        if first is not None and first in rows and first < len(reference_logits):
            ref_row, path_row = reference_logits[first], rows[first]
            ref_top = _top(ref_row)
            entry["at_divergence"] = {
                "position": first,
                "same_prefix": tokens[:first] == reference_tokens[:first],
                "hf_static": ref_top,
                "path": _top(path_row),
                "path_logits_of_reference_top2": [float(path_row[i]) for i in ref_top["ids"][:2]],
                "max_abs_logit_difference": float((ref_row - path_row).abs().max()),
            }
        paths[name] = entry
    prefix_len = min(int(case["bundle_divergence"]), len(reference_tokens))
    prefix = torch.cat(
        [prompt, torch.tensor([reference_tokens[:prefix_len]], dtype=torch.long, device=device)],
        dim=1,
    )
    return {
        "dataset": case["dataset"],
        "prompt_id": case["id"],
        "bundle_divergence": case["bundle_divergence"],
        "bundle_group": case["group"],
        "prompt_tokens": int(prompt.shape[1]),
        "paths": paths,
        "pad_independence_at_bundle_divergence": pad_independence(
            target_module, prefix, session=pad_session
        ),
    }


def _prompt(case: dict[str, Any], tokenizer: Any, device: torch.device) -> torch.Tensor:  # noqa: ANN401
    record = next(item for item in load_prompt_set(case["dataset"]) if item["id"] == case["id"])
    return torch.tensor([_chat_ids(tokenizer, record["text"])], dtype=torch.long, device=device)


def diagnose(
    *,
    target_name: str = _TARGET,
    dtypes: tuple[str, ...] = ("float32", "float16"),
    device: torch.device | None = None,
    cases: tuple[dict[str, Any], ...] = CASES,
) -> dict[str, Any]:
    """Run every case in every dtype; ``fp32_all_match`` is the pass/fail line."""
    from transformers import AutoModelForCausalLM

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _load_tokenizer(target_name)
    runs: list[dict[str, Any]] = []
    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        model = AutoModelForCausalLM.from_pretrained(target_name, torch_dtype=dtype).to(device)
        model.eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.generation_config.repetition_penalty = None
        longest = max(int(_prompt(case, tokenizer, device).shape[1]) for case in cases)
        cache_len = longest + _NEW_TOKENS + 64
        from transformers import StaticCache

        targets = {name: _path_target(model, width, cache_len) for name, _g, width in _PATHS}
        hf_cache = StaticCache(config=model.config, max_cache_len=cache_len)
        pad_session = StaticKvSession(
            model, cache_len, compile_forward=device.type == "cuda", pad_widths=(7,)
        )
        for case in cases:
            run = run_case(
                case,
                model,
                tokenizer,
                device,
                targets=targets,
                hf_cache=hf_cache,
                pad_session=pad_session,
            )
            run["dtype"] = dtype_name
            runs.append(run)
        del model, targets, hf_cache, pad_session
        if device.type == "cuda":
            torch.cuda.empty_cache()
    fp32 = [run for run in runs if run["dtype"] == "float32"]
    from benchmarks.harness import git_commit

    return {
        "git_commit": git_commit(),
        "target": target_name,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "new_tokens": _NEW_TOKENS,
        "cases": runs,
        "fp32_all_match": all(
            path.get("divergence_vs_hf_static") is None
            and path.get("divergence_vs_hf_eager", None) is None
            for run in fp32
            for path in run["paths"].values()
        ),
        "pad_independence_all_bit_identical": all(
            run["pad_independence_at_bundle_divergence"]["row0_bit_identical"] for run in runs
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Write the JSON report and print one line per case and path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default=_TARGET)
    parser.add_argument("--dtypes", default="float32,float16")
    args = parser.parse_args(argv)
    report = diagnose(target_name=args.target, dtypes=tuple(args.dtypes.split(",")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for run in report["cases"]:
        for name, path in run["paths"].items():
            at = path.get("at_divergence") or {}
            print(
                f"{run['dtype']} {run['dataset']}/{run['prompt_id']} {name}: "
                f"divergence vs hf_static {path['divergence_vs_hf_static']}"
                + (
                    f" (ref margin {at['hf_static']['margin_top1_top2']}, "
                    f"path margin {at['path']['margin_top1_top2']}, "
                    f"max |logit diff| {at['max_abs_logit_difference']})"
                    if at
                    else ""
                )
            )
        pad = run["pad_independence_at_bundle_divergence"]
        print(
            f"  pad independence (width {pad['width']}): bit-identical={pad['row0_bit_identical']}"
        )
    print("fp32 all paths match:", report["fp32_all_match"])
    print("pad content never reaches row 0:", report["pad_independence_all_bit_identical"])
    return 0 if report["fp32_all_match"] and report["pad_independence_all_bit_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
