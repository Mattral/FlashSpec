# Fixed prompt sets

These files are the prompt text the harness tokenizes. No acceptance
rate has been measured for them. Do not paste a figure here until a run
in this checkout writes a result JSON.

`benchmarks/configs/` is unchanged. `make bench` globs every YAML in
that directory, including gated Llama configs. Run a real set by passing
`dataset` or `datasets` on a config that names weights you can load,
and a tokenizer for the target model.

## MT-Bench

Ten first-turn questions from LMSYS FastChat,
`fastchat/llm_judge/data/mt_bench/question.jsonl`. Selection: the first
question of each of the eight categories, then the second question of
the first two categories in file order. The first four rows are the
original selection and keep their positions, so `prompts[:4]` is
unchanged.

https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl

| id | category |
| --- | --- |
| 81 | writing |
| 101 | reasoning |
| 111 | math |
| 121 | coding |
| 91 | roleplay |
| 131 | extraction |
| 141 | stem |
| 151 | humanities |
| 82 | writing |
| 92 | roleplay |

License: Apache-2.0. The license text is `LICENSE.apache-2.0` in this
directory. Only the first turn is included. Second turns depend on a
model answer and are omitted. Reference answers and sample code from
that file are omitted. Human-judgment annotations
(`lmsys/mt_bench_human_judgments`, CC-BY-4.0) are not included.

## HumanEval

The ten shortest prompts (by character count) from OpenAI human-eval
`data/HumanEval.jsonl.gz`. The first four rows are the original
selection (the four shortest) and keep their positions:

https://github.com/openai/human-eval

| id | characters |
| --- | --- |
| HumanEval/53 | 115 |
| HumanEval/55 | 130 |
| HumanEval/23 | 133 |
| HumanEval/45 | 138 |
| HumanEval/34 | 149 |
| HumanEval/83 | 164 |
| HumanEval/85 | 167 |
| HumanEval/47 | 168 |
| HumanEval/24 | 172 |
| HumanEval/27 | 179 |

File order is HumanEval/23, HumanEval/45, HumanEval/53, HumanEval/55,
then HumanEval/34, /83, /85, /47, /24, /27.
License: MIT. Copyright (c) OpenAI. The license text is
`LICENSE.humaneval-mit` in this directory.

## HumanEval code edits (`humaneval_edit`)

A copy-heavy set for the prompt-lookup drafter. Each prompt shows one
complete HumanEval function, the task prompt followed by its canonical
solution, and asks for an edit that keeps most of the code. The edits
rotate through three kinds, in this order: rewrite the docstring in Google
style, rename variables to descriptive names, add type hints to local
variables. A good answer restates most of the function.

Selection, fixed before any measurement: tasks in file order whose
canonical solution has at least 5 non-empty lines, skipping the ten tasks
in `humaneval.json` so the sets do not overlap; the first 10. That gives
HumanEval/0, 1, 3, 5, 6, 8, 9, 10, 11 and 12.

Built by `scripts/build_humaneval_edit.py` from `data/HumanEval.jsonl.gz`
at openai/human-eval commit `b9453e8f1865df0a0057a1383767caa400acfbcf`
(file sha256
`b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`).

License: MIT. Copyright (c) OpenAI. The license text is
`LICENSE.humaneval-mit` in this directory. The repository's license was
checked on GitHub (SPDX `MIT`) when the set was built.

A summarisation set was considered first. BillSum was not used: its only
license statement is the `cc0-1.0` tag on the Hugging Face card, whose
licensing section is empty, and the source repository (FiscalNote/BillSum)
declares no license.

## Mixed workload (`mixed`)

The 30 prompts of `mt_bench`, `humaneval` and `humaneval_edit` in one
fixed interleaved order: shuffled once with `random.Random(0)` and stored
in `mixed.json`, so the order does not depend on the Python version.
Record ids are `<dataset>/<id>`. The licenses are those of the three sets.
A benchmark runs the whole list on one engine, so the bandit's state
carries across prompts.

## Alpaca

Stanford Alpaca instruction text is CC BY-NC 4.0. It is not copied into
this tree. `load_prompt_set("alpaca")` raises, and a config that lists
only Alpaca writes no result file. A mixed `datasets` list prints a skip
line and runs the other names.
