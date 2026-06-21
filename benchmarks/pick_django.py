"""Pick N django instances from SWE-bench_Lite and emit:
  - benchmarks/subset.json      (full instances, for later inference)
  - benchmarks/gold_preds.jsonl (predictions = the GOLD patch, for eval-chain test)

Usage: python benchmarks/pick_django.py [N]   (default N=1)

Picking by repo keeps Docker images sharing the django base/env layers (saves
disk). Within django we sort by a cheap "cost" proxy (fewer FAIL_TO_PASS tests +
shorter patch) so the first picks run fastest.
"""

import json
import sys
from pathlib import Path

from datasets import load_dataset

REPO = "django/django"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def _as_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return [v]
    return list(v or [])


def main():
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    django = [x for x in ds if x["repo"] == REPO]
    django.sort(key=lambda x: (len(_as_list(x["FAIL_TO_PASS"])), len(x["patch"])))
    pick = django[:N]

    out = Path(__file__).resolve().parent
    json.dump([dict(x) for x in pick], open(out / "subset.json", "w"))
    with open(out / "gold_preds.jsonl", "w") as f:
        for x in pick:
            f.write(json.dumps({
                "instance_id": x["instance_id"],
                "model_name_or_path": "gold",
                "model_patch": x["patch"],
            }) + "\n")

    print(f"total django in Lite: {len(django)}; picked {len(pick)}:")
    for x in pick:
        print(f"  {x['instance_id']}  | F2P={len(_as_list(x['FAIL_TO_PASS']))}"
              f"  P2P={len(_as_list(x['PASS_TO_PASS']))}  patch={len(x['patch'])} chars")
    print("INSTANCE_IDS=" + ",".join(x["instance_id"] for x in pick))


if __name__ == "__main__":
    main()
