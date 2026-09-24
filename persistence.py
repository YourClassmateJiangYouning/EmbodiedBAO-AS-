"""Crash-safe result persistence helpers.

A scored sweep runs for hours behind Isaac Sim.  Every file it writes can be
interrupted by a crash, an OOM kill, or Ctrl-C, and a *half-written* file is
worse than a missing one:

* ``json.load`` raises on a truncated episode record, so a resume that reads
  one dies before it can re-run that episode;
* a truncated checkpoint used to be swallowed silently, which reset the
  completed set and made the run overwrite episodes that were already
  collected.

Everything here therefore writes to a temporary file in the destination
directory, flushes it to disk, and renames it over the target.  ``os.replace``
is atomic on both NTFS and POSIX filesystems, so any reader sees either the
complete old file or the complete new one -- never a partial write.

The module is deliberately standard-library only: ``main.py`` imports it at
module scope, and the entry point must not pull in Isaac Sim before
``SimulationApp`` has started (see the note at the top of ``main.py``).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

# Characters that are illegal in a Windows path component.  The Linux
# workstation tolerates them, so a tag that works here would fail there (or
# scatter results across directories on either platform).
_ILLEGAL_PATH_CHARS = '<>:"/\\|?*'


def sanitize_tag(tag: Any, fallback: str = "untagged") -> str:
    """Return ``tag`` reduced to a single, filesystem-safe path component.

    A run tag reaches ``results/level{n}/{model}/{tag}/``, ``logs/{tag}/`` and
    the resume checkpoint's filename.  ``--tag a/b`` would otherwise create a
    nested directory and ``--tag run:1`` would fail outright on Windows.
    """
    text = str(tag or "").strip()
    cleaned = "".join(
        "-" if (ch in _ILLEGAL_PATH_CHARS or ord(ch) < 32) else ch for ch in text
    )
    # Windows also rejects names that end in a dot or a space, and "." / ".."
    # would silently escape the results directory.
    cleaned = cleaned.strip(" .")
    if cleaned in ("", ".", ".."):
        return fallback
    return cleaned


def _temporary_path(target: str) -> str:
    directory = os.path.dirname(target)
    base = os.path.basename(target)
    return os.path.join(directory, f".{base}.tmp-{os.getpid()}")


def atomic_write_text(path: str, text: str) -> str:
    """Write ``text`` to ``path`` so no reader ever observes a partial file.

    The temporary file is created in the *destination* directory: ``os.replace``
    is only atomic within one filesystem, and ``/tmp`` is frequently a different
    one from the results volume.
    """
    target = os.path.abspath(path)
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = _temporary_path(target)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return target


def atomic_write_json(path: str, payload: Any, indent: int = 2) -> str:
    """Atomically serialise ``payload`` as UTF-8 JSON."""
    return atomic_write_text(
        path, json.dumps(payload, indent=indent, ensure_ascii=False)
    )


def backup_corrupt_file(path: str) -> Optional[str]:
    """Move an unreadable file aside so the loss is visible instead of silent.

    Returns the backup path, or ``None`` when there was nothing to move or the
    move failed.  Renaming (rather than deleting) keeps the damaged bytes
    available for inspection while still allowing the run to make progress.
    """
    if not os.path.exists(path):
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = f"{path}.corrupt-{stamp}"
    counter = 1
    while os.path.exists(backup):
        counter += 1
        backup = f"{path}.corrupt-{stamp}-{counter}"
    try:
        os.replace(path, backup)
    except OSError:
        return None
    return backup


def load_json_file(path: str) -> Any:
    """Read UTF-8 JSON, raising on any decoding or IO problem."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
