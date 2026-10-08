"""Check that effective_tag is idempotent for every roster entry, including the two cases that
broke it: a model id containing a slash, and a model whose request parameters append a suffix
after the scene suffix.

    python tools/check_tags.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import effective_tag  # noqa: E402


def roster() -> list:
    import json

    with open("models.json", encoding="utf-8") as handle:
        return [entry["runner"] for entry in json.load(handle)["models"]]


def main() -> int:
    bad = 0
    for model in roster():
        for scene in ("", "stage1.1", "stage1.2"):
            once = effective_tag(model, "", scene)
            twice = effective_tag(model, once, scene)
            thrice = effective_tag(model, twice, scene)
            stable = once == twice == thrice
            if not stable:
                bad += 1
                print(f"UNSTABLE model={model!r} scene={scene!r}")
                print(f"    once  : {once}")
                print(f"    twice : {twice}")
                print(f"    thrice: {thrice}")
            if scene and scene not in once:
                bad += 1
                print(f"MISSING SCENE model={model!r} scene={scene!r} -> {once}")
            if once.count("stage1.") > 1:
                bad += 1
                print(f"DUPLICATED SCENE model={model!r} scene={scene!r} -> {once}")
            if once.count("v7-state-axes") > 1:
                bad += 1
                print(f"DUPLICATED PROTOCOL model={model!r} scene={scene!r} -> {once}")
        print(f"  {model:<34} -> {effective_tag(model, '', 'stage1.1')}")
    print()
    print(f"{bad} problem(s)" if bad else "all roster entries are stable and single-suffixed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
