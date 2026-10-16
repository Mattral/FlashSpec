# Fixed prompt sets

These files are the prompt text the harness tokenizes. No acceptance
rate has been measured for them. Do not paste a figure here until a run
in this checkout writes a result JSON.

`benchmarks/configs/` is unchanged. `make bench` globs every YAML in
that directory, including gated Llama configs. Run a real set by passing
`dataset` or `datasets` on a config that names weights you can load,
and a tokenizer for the target model.

## MT-Bench

Four first-turn questions from LMSYS FastChat,
`fastchat/llm_judge/data/mt_bench/question.jsonl`:

https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl

| id | category |
| --- | --- |
| 81 | writing |
| 101 | reasoning |
| 111 | math |
| 121 | coding |

License: Apache-2.0. The license text is `LICENSE.apache-2.0` in this
directory. Only the first turn is included. Second turns depend on a
model answer and are omitted. Reference answers and sample code from
that file are omitted. Human-judgment annotations
(`lmsys/mt_bench_human_judgments`, CC-BY-4.0) are not included.

## HumanEval

Four short prompts from OpenAI human-eval:

https://github.com/openai/human-eval

| id | characters |
| --- | --- |
| HumanEval/53 | 115 |
| HumanEval/55 | 130 |
| HumanEval/23 | 133 |
| HumanEval/45 | 138 |

File order is HumanEval/23, HumanEval/45, HumanEval/53, HumanEval/55.
License: MIT. Copyright (c) OpenAI. The license text is
`LICENSE.humaneval-mit` in this directory.

## Alpaca

Stanford Alpaca instruction text is CC BY-NC 4.0. It is not copied into
this tree. `load_prompt_set("alpaca")` raises, and a config that lists
only Alpaca writes no result file. A mixed `datasets` list prints a skip
line and runs the other names.
