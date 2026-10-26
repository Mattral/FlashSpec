"""The README cannot rot: its code runs, its links resolve, its API table matches the code.

- Every Python block in ``README.md`` and ``docs/examples.md`` is executed,
  in order, in one namespace per file. On CPU the hub checkpoints are
  replaced by tiny random Llamas and a character tokenizer (the code is
  the README's own; only the weights differ). The same blocks run against
  the real checkpoints in ``test_python_blocks_with_real_checkpoints``,
  which needs CUDA and is skipped elsewhere.
- The custom drafter from the "Use your own drafter" recipe is checked for
  losslessness with the suite's chi-squared rule.
- Every server command parses with the server's own parser, and every
  request body validates against the server's request model.
- Every link and image is an absolute https URL (PyPI breaks relative
  ones), points at a file that exists, and every anchor names a heading.
- The "API at a glance" block equals ``scripts/readme_api.py``'s render,
  and both generated result blocks keep their markers.
- ``llms.txt`` follows the llmstxt.org layout and its links resolve.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import re
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

from scripts.ingest_gpu_bundle import BEGIN, END, STATUS_BEGIN, STATUS_END
from scripts.readme_api import BEGIN as API_BEGIN
from scripts.readme_api import END as API_END
from scripts.readme_api import render as render_api

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
EXAMPLES = ROOT / "docs" / "examples.md"
LLMS = ROOT / "llms.txt"
REPO = "https://github.com/Mattral/FlashSpec"
_VOCAB = 128
_CHI2_CRITICAL_DF5 = 15.086


def _blocks(path: Path, language: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(rf"^```{language}\n(.*?)^```", text, flags=re.S | re.M)


def _section(text: str, heading: str) -> str:
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else len(text)]


# ── tiny stand-ins for the hub checkpoints ───────────────────────────────────


class _CharTokenizer:
    """One token per character; enough for the README's encode/decode calls."""

    def __len__(self) -> int:
        return _VOCAB

    def __call__(self, text: str, return_tensors: str = "pt") -> Any:  # noqa: ANN401
        ids = torch.tensor([[ord(ch) % _VOCAB for ch in text]], dtype=torch.long)
        return type("Encoding", (), {"input_ids": ids})()

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:  # noqa: ANN401
        values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "".join(chr(32 + int(v) % 95) for v in values)


def _tiny(seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        initializer_range=0.1,
    )
    return LlamaForCausalLM(config).eval()


@pytest.fixture
def tiny_hub(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Patch from_pretrained to tiny local models and force CPU; yield the names loaded."""
    loaded: list[str] = []
    cache: dict[str, LlamaForCausalLM] = {}

    def model(name: str, *args: Any, torch_dtype: Any = None, **kwargs: Any) -> Any:  # noqa: ANN401
        loaded.append(name)
        if name not in cache:
            cache[name] = _tiny(len(cache) + 1)
        return cache[name].to(torch_dtype or torch.float32)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _CharTokenizer()
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    logger = logging.getLogger("flashspec")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield loaded
    logger.setLevel(previous)


def _run(path: Path) -> tuple[dict[str, Any], str]:
    namespace: dict[str, Any] = {"__name__": "readme"}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for block in _blocks(path, "python"):
            exec(compile(block, str(path), "exec"), namespace)  # noqa: S102
    return namespace, out.getvalue()


# ── code ─────────────────────────────────────────────────────────────────────


def test_readme_python_blocks_run_on_cpu(tiny_hub: list[str]) -> None:
    """Quickstart and every recipe, in order, on tiny models."""
    namespace, printed = _run(README)
    assert set(tiny_hub) == {"Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"}
    assert "tok/s, acceptance" in printed
    # The last recipe's engine ran and produced every token.
    assert namespace["result"].n_tokens_generated == 32
    # The drafter recipe built the composite strategy's arm and a lookup-only engine.
    assert "label='composite-g6-f2', fallback=0, fallback_gamma=2" in printed
    assert namespace["lookup"].arms[0].fallback is None
    assert "False |" in printed  # the compile-off recipe printed its decision


def test_examples_page_runs_on_cpu(tiny_hub: list[str]) -> None:
    """docs/examples.md, on tiny models."""
    namespace, _printed = _run(EXAMPLES)
    assert set(tiny_hub) == {"Felladrin/Llama-160M-Chat-v1", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
    assert namespace["result"].n_tokens_generated == 16


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real checkpoints need CUDA")
def test_python_blocks_with_real_checkpoints() -> None:
    """The README and examples blocks unpatched: downloads the real checkpoints."""
    _namespace, printed = _run(README)
    assert "tok/s" in printed
    _run(EXAMPLES)


def test_quickstart_is_short() -> None:
    """The quickstart is one copy-paste block of at most 20 lines."""
    quickstart = _blocks(README, "python")[0]
    assert "engine.generate" in quickstart
    assert len(quickstart.splitlines()) <= 20


def test_custom_drafter_recipe_is_lossless(tiny_hub: list[str]) -> None:
    """RepeatLast (one-hot q) with Algorithm 1: the first token follows the target."""
    from flashspec import BanditConfig, FlashSpecConfig, SamplingConfig, SpeculativeEngine
    from flashspec.bandit import UCB1Selector
    from flashspec.engine import TargetModel

    namespace, _ = _run(README)
    drafter_class = namespace["RepeatLast"]
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=6,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        initializer_range=0.3,
    )
    model = LlamaForCausalLM(config).eval()
    target = TargetModel("toy", torch.device("cpu"), torch.float32, model=model)
    engine = SpeculativeEngine(
        FlashSpecConfig(
            device="cpu",
            max_new_tokens=2,
            bandit=BanditConfig(n_arms=1),
            sampling=SamplingConfig(temperature=1.0),
        ),
        [drafter_class(6)],
        target,
        UCB1Selector(n_arms=1),
    )
    prompt = torch.tensor([[1, 2, 3]])
    torch.manual_seed(5)
    n = 3000
    first = torch.cat([engine.generate(prompt).output_ids[:, :1] for _ in range(n)]).view(-1)
    with torch.no_grad():
        p = torch.softmax(model(prompt).logits[0, -1], dim=-1)
    counts = torch.bincount(first, minlength=6).float()
    chi2 = float((((counts - n * p) ** 2) / (n * p)).sum())
    assert chi2 < _CHI2_CRITICAL_DF5, chi2


def _commands(path: Path) -> list[list[str]]:
    out = []
    for block in _blocks(path, "bash"):
        for command in block.replace("\\\n", " ").splitlines():
            if command.strip() and not command.lstrip().startswith("#"):
                out.append(shlex.split(command))
    return out


def test_server_commands_parse_and_requests_validate() -> None:
    """Every `python -m flashspec.serve` line parses; every curl body is a valid request."""
    from flashspec.serve.__main__ import build_parser
    from flashspec.serve.app import ChatRequest, CompletionRequest

    commands = _commands(README)
    serve = [c for c in commands if c[:3] == ["python", "-m", "flashspec.serve"]]
    curls = [c for c in commands if c and c[0] == "curl"]
    assert serve and curls
    for command in serve:
        build_parser().parse_args(command[3:])
    for command in curls:
        url = next(part for part in command if part.startswith("http"))
        body = json.loads(command[command.index("-d") + 1])
        model = ChatRequest if url.endswith("/chat/completions") else CompletionRequest
        model.model_validate(body)


# ── links, markup, generated blocks ──────────────────────────────────────────


def _slug(heading: str) -> str:
    """GitHub's heading anchor."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
    return {_slug(m) for m in re.findall(r"^#{1,6} (.+)$", text, flags=re.M)}


def _links(text: str) -> list[str]:
    text = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
    return re.findall(r"\]\(([^)\s]+)\)", text)


def _check_link(url: str) -> None:
    assert url.startswith("https://"), f"not an absolute https URL: {url}"
    for prefix in (f"{REPO}/blob/main/", f"{REPO}/tree/main/"):
        if url.startswith(prefix):
            path, _, anchor = url.removeprefix(prefix).partition("#")
            assert (ROOT / path).exists(), f"missing file for {url}"
            if anchor:
                assert anchor in _anchors(ROOT / path), f"missing anchor in {url}"
    raw = "https://raw.githubusercontent.com/Mattral/FlashSpec/main/"
    if url.startswith(raw):
        assert (ROOT / url.removeprefix(raw)).exists(), url
    if url.startswith(f"{REPO}#"):
        assert url.split("#", 1)[1] in _anchors(README), f"missing README anchor: {url}"


def test_readme_links_are_absolute_and_resolve() -> None:
    """PyPI renders this README: no relative links, and every target exists."""
    links = _links(README.read_text(encoding="utf-8"))
    assert len(links) > 20
    for url in links:
        _check_link(url)


def test_readme_has_no_raw_html_and_few_badges() -> None:
    """No HTML that PyPI strips; 4 to 6 badges, all on one block under the title."""
    text = README.read_text(encoding="utf-8")
    without_comments = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    without_code = re.sub(r"^```.*?^```", "", without_comments, flags=re.S | re.M)
    assert not re.search(r"<\s*(p|img|div|br|a|h\d|picture|source)\b", without_code, re.I)
    assert "align=" not in without_code
    badges = re.findall(r"^\[!\[", text, flags=re.M)
    assert 4 <= len(badges) <= 6


def test_generated_blocks_are_present_once() -> None:
    """The ingest's two result blocks and the API block keep their markers."""
    text = README.read_text(encoding="utf-8")
    for marker in (BEGIN, END, STATUS_BEGIN, STATUS_END, API_BEGIN, API_END):
        assert text.count(marker) == 1, marker
    assert text.index(STATUS_BEGIN) < text.index(STATUS_END)
    assert text.index(BEGIN) < text.index(END)


def test_api_block_matches_the_code() -> None:
    """Names, signatures and defaults come from the package; regenerate when this fails."""
    text = README.read_text(encoding="utf-8")
    block = text[text.index(API_BEGIN) + len(API_BEGIN) : text.index(API_END)]
    assert block == "\n" + render_api(), "run: python scripts/readme_api.py"


def test_readme_order_follows_the_audiences() -> None:
    """Decide in 30 seconds, try in 5 minutes, then reference."""
    text = README.read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, flags=re.M)
    assert headings == [
        "What it is",
        "When to use it",
        "Install",
        "Quickstart",
        "Recipes",
        "API at a glance",
        "Results",
        "How it works",
        "Correctness",
        "Status and limitations",
        "Citing",
        "Contributing",
        "License",
    ]
    how = _section(text, "How it works").strip().splitlines()[1:]
    assert len([line for line in how if line.strip()]) <= 8


def test_llms_txt() -> None:
    """llmstxt.org layout: H1, a blockquote summary, H2 link lists; links resolve."""
    text = LLMS.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# FlashSpec"
    assert any(line.startswith("> ") for line in lines[1:4])
    sections = re.findall(r"^## (.+)$", text, flags=re.M)
    assert {"Docs", "Examples"} <= set(sections)
    listed = text[text.index("\n## ") :].splitlines()
    for line in listed:
        if line.startswith("- "):
            assert re.match(r"- \[[^\]]+\]\(https://[^)]+\)", line), line
    for url in _links(text):
        _check_link(url)


def test_serve_extra_covers_the_server_install_line() -> None:
    """`pip install "flashspec[serve]"` is enough: the server loads with device_map (accelerate)."""
    import tomllib

    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    names = {re.split(r"[<>=;\s\[]", item, maxsplit=1)[0] for item in extras["serve"]}
    assert {"fastapi", "uvicorn", "accelerate"} <= names
    installs = [c for c in _commands(README) if c[:2] == ["pip", "install"]]
    assert ["pip", "install", "flashspec[serve]"] in installs


def test_throughput_chart_matches_the_status_table() -> None:
    """The README chart's image exists, and every value it plots is a cell of the status table."""
    from scripts.readme_chart import chart_rows

    text = README.read_text(encoding="utf-8")
    images = re.findall(
        r"!\[[^\]]*\]\((https://raw[^)]+/docs/images/throughput_(\w+)\.png)\)", text
    )
    assert len(images) == 1
    _url, commit = images[0]
    assert (ROOT / "docs" / "images" / f"throughput_{commit}.png").exists()
    status = text[text.index(STATUS_BEGIN) : text.index(STATUS_END)]
    assert f"git `{commit}`" in status
    rows = chart_rows(ROOT / "benchmarks" / "results" / "gpu" / commit / "results")
    assert len(rows) == 4
    for _label, values in rows:
        flashspec, compiled_hf, eager_hf = values
        for value in (flashspec, compiled_hf, eager_hf):
            assert f"| {value} |" in status or f" {value} |" in status, value


_IMAGE = "ghcr.io/mattral/flashspec"


def test_container_commands_name_this_version_and_parse() -> None:
    """Every documented image reference is this release's tag, and server args parse.

    Arguments after the image replace its default command, so they must
    include ``--host 0.0.0.0`` or the server is unreachable from outside.
    """
    import flashspec
    from flashspec.serve.__main__ import build_parser

    files = [
        README,
        ROOT / "docs" / "install.md",
        ROOT / "deploy" / "docker-compose.yml",
        ROOT / "deploy" / "k8s" / "flashspec-deployment.yaml",
    ]
    for path in files:
        tags = re.findall(rf"{re.escape(_IMAGE)}:([\w.\-]+)", path.read_text(encoding="utf-8"))
        assert tags, path
        assert set(tags) == {flashspec.__version__}, (path, tags)
    runs = [c for c in _commands(README) if c[:2] == ["docker", "run"]]
    assert runs
    for command in runs:
        image_at = next(i for i, part in enumerate(command) if part.startswith(_IMAGE))
        args = command[image_at + 1 :]
        if args:
            parsed = build_parser().parse_args(args)
            assert parsed.host == "0.0.0.0", command


def test_quickstart_auto_runs_composite_and_matches_hf_greedy(tiny_hub: list[str]) -> None:
    """No BanditConfig: the arm count follows the drafters (composite-g6-f2, HF-exact greedy)."""
    quickstart = _blocks(README, "python")[0]
    assert "BanditConfig" not in quickstart
    namespace: dict[str, Any] = {"__name__": "readme"}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(quickstart, "README quickstart", "exec"), namespace)  # noqa: S102
    engine, result = namespace["engine"], namespace["result"]
    assert engine.drafter_strategy == "composite-g6-f2"
    assert "auto: one model drafter" in engine.drafter_strategy_reason
    assert engine.arms[0].gamma == 6 and engine.arms[0].fallback_gamma == 2
    assert "drafter strategy: composite-g6-f2" in out.getvalue()
    target_module = namespace["target"]._model
    prompt = namespace["prompt"]
    with torch.no_grad():
        reference = target_module.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=128,
            min_new_tokens=128,
            eos_token_id=None,
            pad_token_id=0,
        )[:, prompt.shape[1] :]
    assert result.n_tokens_generated == 128
    assert torch.equal(result.output_ids[:, :128], reference)
