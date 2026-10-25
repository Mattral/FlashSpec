# Examples

The README quickstart and recipes are the main examples; a test runs
every Python block in the README and on this page, so they stay current.
The notebooks are in
[`notebooks/`](https://github.com/Mattral/FlashSpec/tree/main/notebooks).

## A CPU example with `load_drafters` (TinyLlama)

This was the quickstart before 0.3.0. It loads the drafter through
`load_drafters` (the registry path) instead of `HuggingFaceDrafter`, and
samples at temperature 1.0 on CPU in float32.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flashspec import FlashSpecConfig, BanditConfig, SamplingConfig, TRITON_AVAILABLE
from flashspec.bandit import UCB1Selector
from flashspec.engine import SpeculativeEngine, TargetModel, load_drafters

print(f"Triton kernels available: {TRITON_AVAILABLE}")

# Public checkpoints. Both report vocab_size 32000.
# Llama-3 reports vocab_size 128256. Do not pair it with this drafter.
drafter_id = "Felladrin/Llama-160M-Chat-v1"
target_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

config = FlashSpecConfig(
    device="cpu",
    dtype="float32",
    drafter_name=drafter_id,
    target_name=target_id,
    bandit=BanditConfig(n_arms=1, strategy="ucb1"),
    sampling=SamplingConfig(gamma=4, temperature=1.0),
    max_new_tokens=16,
)
# TargetModel(hub_id, device) passes device_map and needs accelerate,
# which this package does not depend on. Load the weights here and inject
# them, which is what benchmarks/harness.py does.
drafter_module = AutoModelForCausalLM.from_pretrained(
    drafter_id, torch_dtype=torch.float32
).to(config.device)
target_module = AutoModelForCausalLM.from_pretrained(
    target_id, torch_dtype=torch.float32
).to(config.device)
drafter_module.eval()
target_module.eval()
drafters = load_drafters(
    [drafter_id],
    device=config.device,
    dtype=torch.float32,
    temperature=config.sampling.temperature,
    models={drafter_id: drafter_module},
)
target = TargetModel(
    target_id,
    torch.device(config.device),
    torch.float32,
    model=target_module,
)
engine = SpeculativeEngine(
    config,
    drafters,
    target,
    UCB1Selector(n_arms=1, window_size=config.bandit.window_size),
)
tokenizer = AutoTokenizer.from_pretrained(target_id)
prompt = tokenizer("The capital of France is", return_tensors="pt").input_ids
result = engine.generate(prompt, max_new_tokens=16)
new_ids = result.output_ids[0]
new_ids = new_ids[new_ids != -1]
print(tokenizer.decode(new_ids, skip_special_tokens=True))
```

Both checkpoints are public. `benchmarks/harness.py` refuses a drafter and a target whose vocabulary sizes differ and writes no result file. `SpeculativeEngine` itself does not apply that check. `notebooks/01_quickstart.py` is a separate CPU toy: the names are `"toy"`, the log-probs are random, and it does not download these checkpoints. With one drafter the bandit cannot adapt, and the loader says so.
