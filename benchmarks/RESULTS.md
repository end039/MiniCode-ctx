# MiniCode on SWE-bench Lite (django subset)

**Result: 27 / 30 resolved (90%)** on the 30 easiest django instances of
SWE-bench Lite, using MiniCode (this fork) driving `deepseek-v4-pro`.

## Setup
- **Agent**: MiniCode with container-aware execution (all file/command tools run
  inside each instance's Docker container at `/testbed`), intra-turn context
  compaction (256k window), and a read-only exploration sub-agent. Background
  memory disabled for eval.
- **Model**: `deepseek-v4-pro` via the Anthropic-compatible endpoint.
- **Inference**: `benchmarks/swe_bench_runner.py` — for each instance, start its
  SWE-bench container, run the agent, extract `git diff` → `predictions.jsonl`.
- **Scoring**: the official `swebench.harness.run_evaluation` (predictions +
  gold test_patch applied in clean containers).
- **Subset selection**: `benchmarks/pick_django.py 30` — the 30 django instances
  with the fewest FAIL_TO_PASS tests + shortest gold patch (i.e. the easiest;
  the headline number is therefore optimistic vs. full SWE-bench Lite).

## Reproduce
```bash
source deepseek.env
python benchmarks/pick_django.py 30
python benchmarks/swe_bench_runner.py --parallel 1 --keep-images
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path benchmarks/predictions.jsonl \
  --run_id minicode-django30 --cache_level instance --max_workers 4
```

## Unresolved (3)
`django__django-12113`, `django__django-15320`, `django__django-15695`

## Notes
- Run serial (`--parallel 1`): WSL was memory-limited; two django test suites
  in parallel caused OOM. With ≥12GB WSL memory, `--parallel 2` is viable.
- All 30 produced non-empty, applicable patches (0 empty, 0 harness errors).
