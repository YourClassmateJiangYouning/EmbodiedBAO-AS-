"""Dry-run the result-saving pipeline without Isaac Sim or an API key.

Writes a few episodes through the real runner and the real CSV exporter, then
prints every artifact produced and the field list of the episode and step
records.  This is a check that the on-disk contract the paper depends on is
actually what gets written -- the JSON tree per tag, the per-step sidecar, the
logs, the tag-qualified CSV -- and it can be run before committing hours to a
real sweep.

    python tools/dry_run_save.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        import main as main_module
        import test_bao_integration as harness

        tmp = os.path.join("_test_tmp", "dryrun")
        harness.shutil.rmtree("_test_tmp", ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)

        harness._install_scripted_agent("frontal")
        try:
            env = harness.FakeBAOEnv()
            runner = harness.BAOExperimentRunner(
                env=env,
                model="dry-model",
                max_steps=harness.DEFAULT_MAX_STEPS,
                results_root=os.path.join(tmp, "results"),
                logs_root=os.path.join(tmp, "logs"),
                tag="drytag",
            )
            episodes = runner.run_level(level=4, episodes=3)
            csv_path = main_module.save_episodes_csv(
                episodes,
                model="dry-model",
                level=4,
                results_root=os.path.join(tmp, "results"),
                timestamp="T",
                tag="drytag",
            )
        finally:
            harness._restore_agent_adapter()

        print("=== artifacts produced ===")
        for pattern in ("results/**/*.json", "results/**/*.csv", "logs/**/*.txt"):
            for path in sorted(glob.glob(os.path.join(tmp, pattern), recursive=True)):
                print(
                    "  %-70s %7d bytes"
                    % (os.path.relpath(path, tmp), os.path.getsize(path))
                )

        print()
        print("=== episode record fields ===")
        episode_path = os.path.join(
            tmp, "results/level4/dry-model/drytag/episode_000.json"
        )
        episode = json.load(open(episode_path, encoding="utf-8"))
        print("  " + ", ".join(sorted(k for k in episode if k != "steps")))
        print("=== step record fields ===")
        print("  " + ", ".join(sorted(episode["steps"][0].keys())))

        print()
        print("=== csv name and header ===")
        print("  " + os.path.basename(csv_path))
        with open(csv_path, encoding="utf-8") as handle:
            print("  " + handle.readline().strip())

        harness.shutil.rmtree("_test_tmp", ignore_errors=True)
        return 0
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    raise SystemExit(main())
