"""SWE-bench *inference* runner for MiniCode.

For each instance: start its SWE-bench container (repo at /testbed), run the
MiniCode agent INSIDE it (container-aware tools), extract the agent's git diff,
and append it to predictions.jsonl. Scoring is intentionally NOT done here —
hand predictions.jsonl to the official `swebench.harness.run_evaluation`.

Config (per the project's Phase-2 decisions):
  - context compaction ON (window 256k; tests our compactor even though
    DeepSeek supports 1M)
  - background memory OFF (irrelevant to solving an isolated issue; we simply
    don't instantiate it, and set the env flag as belt-and-suspenders)
  - explore sub-agent available, and now propagates the container
  - inference parallelism configurable (default 2)

Usage:
  source deepseek.env
  python benchmarks/swe_bench_runner.py --instance-ids django__django-14411
  python benchmarks/swe_bench_runner.py            # uses benchmarks/subset.json
  python benchmarks/swe_bench_runner.py --parallel 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make minicode importable when run as `python benchmarks/swe_bench_runner.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MINI_CODE_BACKGROUND_MEMORY", "0")  # off for eval

from datasets import load_dataset  # noqa: E402
from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: E402

from minicode.agent_loop import run_agent_turn  # noqa: E402
from minicode.anthropic_adapter import AnthropicModelAdapter  # noqa: E402
from minicode.config import load_runtime_config  # noqa: E402
from minicode.context_compactor import ContextCompactor  # noqa: E402
from minicode.subagent import SubAgentTracker  # noqa: E402
from minicode.tools import create_default_tool_registry  # noqa: E402

TESTBED = "/testbed"
# All SWE-bench images put the working conda env here; make `sh -lc` use it so
# the agent's `python`/test commands hit the right interpreter + deps.
ENV_INJECT = "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH"
CONTEXT_WINDOW = 256_000
MAX_STEPS = 60

SWE_SYSTEM_PROMPT = """You are mini-code, an autonomous coding agent fixing one real bug in a Python repository.

The repository is checked out at /testbed in this environment. ALL your file and
command tools already operate INSIDE the container at /testbed: read_file,
write_file, edit_file, patch_file, grep_files, list_files, run_command (and
dispatch_agent for read-only exploration) all run there.

How to work:
1. Read the issue. Locate the root cause by exploring /testbed (grep_files /
   read_file, or dispatch_agent for a broad "where/how" search).
2. Edit the SOURCE under /testbed to fix it. Do NOT add or modify tests — the
   grader supplies its own tests.
3. Keep the change minimal and targeted at the issue.
4. You may run the project's own tests to check your fix, e.g. for Django:
   run_command  python tests/runtests.py <dotted.test.module> --parallel=1

Response protocol:
- While still working (more tool calls coming), begin your message with <progress>.
- When the fix is complete, begin your message with <final> and briefly summarize
  what you changed and why.
- Never stop after a <progress>; keep going until the fix is done, then <final>.
- Do NOT run git or commit — the harness extracts your diff automatically.
"""


def _sh(argv: list[str], timeout: int = 120, stdin: str | None = None) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            argv, input=stdin, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"


def docker_start(image: str, name: str) -> None:
    _sh(["docker", "rm", "-f", name])
    # `docker run` auto-pulls if the image is absent; allow time for that.
    code, _out, err = _sh(
        ["docker", "run", "-d", "--name", name, image, "sleep", "infinity"],
        timeout=1800,
    )
    if code != 0:
        raise RuntimeError(f"docker run failed for {image}: {err.strip()}")
    # Make the testbed conda env the default for the agent's `sh -lc` commands.
    _sh(["docker", "exec", name, "bash", "-c", f'echo "{ENV_INJECT}" >> /etc/profile'])


def docker_rm(name: str) -> None:
    _sh(["docker", "rm", "-f", name])


def docker_rmi(image: str) -> None:
    # Bound disk: each django instance image (base+env+instance) is ~4GB and
    # django versions rarely share env, so keeping all 30 would be ~120GB.
    _sh(["docker", "rmi", "-f", image], timeout=120)


def extract_patch(name: str) -> str:
    # `add -A` so new files are included; diff against HEAD (the base_commit).
    _sh(["docker", "exec", name, "git", "-C", TESTBED, "add", "-A"])
    _code, out, _err = _sh(
        ["docker", "exec", name, "git", "-C", TESTBED, "diff", "--cached", "HEAD"],
        timeout=120,
    )
    return out


def solve_instance(inst: dict, model, model_label: str, tools, idx: int, total: int,
                   cleanup_image: bool = True) -> dict:
    iid = inst["instance_id"]
    image = make_test_spec(inst, namespace="swebench").instance_image_key
    name = f"minicode-swe-{iid}".replace("__", "_")
    t0 = time.time()
    print(f"[{idx}/{total}] {iid}: starting container ({image})", flush=True)
    docker_start(image, name)
    try:
        compactor = ContextCompactor(
            model_adapter=model, model_name=model_label, window=CONTEXT_WINDOW
        )
        messages = [
            {"role": "system", "content": SWE_SYSTEM_PROMPT},
            {"role": "user", "content":
                f"[Issue in {inst['repo']}]\n\n{inst['problem_statement']}\n\n"
                f"Fix the bug in the source under /testbed."},
        ]
        run_agent_turn(
            model=model,
            tools=tools,
            messages=messages,
            cwd=TESTBED,
            permissions=None,         # auto-allow: the container is the sandbox
            compactor=compactor,
            tool_container=name,
            max_steps=MAX_STEPS,
        )
        patch = extract_patch(name)
        print(f"[{idx}/{total}] {iid}: done in {time.time()-t0:.0f}s, "
              f"patch={len(patch)} chars{' (EMPTY!)' if not patch.strip() else ''}",
              flush=True)
        return {"instance_id": iid,
                "model_name_or_path": model_label,
                "model_patch": patch}
    finally:
        docker_rm(name)
        if cleanup_image:
            docker_rmi(image)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-ids", default=None,
                    help="comma-separated; default = benchmarks/subset.json")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--out", default="benchmarks/predictions.jsonl")
    ap.add_argument("--keep-images", action="store_true",
                    help="keep each instance image after solving (default: remove, "
                         "to bound disk — django images are ~4GB each, rarely shared)")
    args = ap.parse_args()

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    by_id = {x["instance_id"]: x for x in ds}
    if args.instance_ids:
        ids = [s for s in args.instance_ids.split(",") if s]
    else:
        subset = json.loads(Path("benchmarks/subset.json").read_text())
        ids = [x["instance_id"] for x in subset]
    insts = [by_id[i] for i in ids]

    runtime = load_runtime_config(str(Path.cwd()))
    model_label = f"minicode-{runtime['model']}"
    tracker = SubAgentTracker()
    # cwd=TESTBED so the dispatch tool's context.cwd is the in-container repo.
    tools = create_default_tool_registry(TESTBED, runtime=runtime, subagent_tracker=tracker)
    model = AnthropicModelAdapter(runtime, tools)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["instance_id"])
    todo = [x for x in insts if x["instance_id"] not in done]
    print(f"instances: {len(insts)} total, {len(done)} already done, "
          f"{len(todo)} to run, parallel={args.parallel}, model={runtime['model']}",
          flush=True)

    lock = threading.Lock()

    def work(pair):
        i, inst = pair
        try:
            pred = solve_instance(inst, model, model_label, tools, i + 1, len(todo),
                                  cleanup_image=not args.keep_images)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {inst['instance_id']}: {e}", flush=True)
            return None
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(pred) + "\n")
        return pred["instance_id"]

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        for iid in ex.map(work, list(enumerate(todo))):
            if iid:
                print(f"  + wrote prediction: {iid}", flush=True)

    print(f"DONE. predictions -> {out}", flush=True)


if __name__ == "__main__":
    main()
