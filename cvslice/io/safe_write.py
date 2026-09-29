"""Stage clip outputs before replacing live files; roll back handled failures.

This protects against cancellation and ordinary write errors, not power loss
during a multi-file commit. Temporary files share the destination filesystem.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path


def _temporary(target, suffix=None):
    target = Path(target)
    fd, path = tempfile.mkstemp(prefix=".cvslice-", suffix=suffix or target.suffix,
                                dir=target.parent)
    os.close(fd)
    return path


def atomic_json(path, value):
    tmp = _temporary(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class StagedWrites:
    def __init__(self):
        self.entries = []

    def __enter__(self):
        return self

    def stage(self, target, *, backup=False):
        target = os.path.abspath(target)
        if any(os.path.normcase(p) == os.path.normcase(target) for p, _, _ in self.entries):
            raise ValueError(f"Duplicate output: {target}")
        tmp = _temporary(target)
        self.entries.append((target, tmp, backup))
        return tmp

    def commit(self):
        rollback = {}
        replaced = []
        keep = set()
        try:
            # Prepare ALL recovery copies before touching any destination.
            for target, staged, backup in self.entries:
                if not os.path.isfile(staged) or os.path.getsize(staged) == 0:
                    raise OSError(f"Missing/empty output: {target}")
                if os.path.exists(target):
                    previous = _temporary(target, ".rollback")
                    rollback[target] = previous
                    shutil.copy2(target, previous)
                    if backup and not os.path.exists(target + ".bak"):
                        backup_tmp = _temporary(target, ".backup")
                        try:
                            shutil.copy2(target, backup_tmp)
                            os.replace(backup_tmp, target + ".bak")
                        finally:
                            if os.path.exists(backup_tmp):
                                os.remove(backup_tmp)
                else:
                    rollback[target] = None
            for target, staged, _ in self.entries:
                os.replace(staged, target)
                replaced.append(target)
        except Exception as error:
            failures = []
            for target in reversed(replaced):
                try:
                    previous = rollback[target]
                    if previous is None:
                        os.remove(target)
                    else:
                        os.replace(previous, target)
                except OSError as exc:
                    if rollback[target]:
                        keep.add(rollback[target])
                    failures.append(f"{target}: {exc}; recovery={rollback[target]}")
            if failures:
                raise OSError("Rollback incomplete: " + "; ".join(failures)) from error
            raise
        finally:
            for previous in rollback.values():
                if previous and previous not in keep and os.path.exists(previous):
                    os.remove(previous)

    def __exit__(self, *args):
        for _, staged, _ in self.entries:
            if os.path.exists(staged):
                os.remove(staged)
