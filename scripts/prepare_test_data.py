"""Prepare reproducible local databases used by the regression suite.

The Chinook file is pinned to an exact upstream Git commit and verified with
SHA-256.  The Olist fixture is generated locally with a fixed random seed.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHINOOK_COMMIT = "7f67772503d71ba90f19283c38e93923addb43fa"
CHINOOK_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/"
    f"{CHINOOK_COMMIT}/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
)
CHINOOK_SHA256 = "7651ba378ac2fcd0dfc3c66fb101f7a7eed3ba39a612ec642b96e20702061f15"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_chinook() -> None:
    target = DATA_DIR / "chinook.sqlite"
    if target.exists() and sha256(target) == CHINOOK_SHA256:
        print("Chinook fixture is ready.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".sqlite.download")
    urllib.request.urlretrieve(CHINOOK_URL, temporary)
    actual = sha256(temporary)
    if actual != CHINOOK_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Chinook SHA-256 mismatch: expected {CHINOOK_SHA256}, got {actual}"
        )
    temporary.replace(target)
    print("Chinook fixture downloaded and verified.")


def prepare_olist() -> None:
    target = DATA_DIR / "olist.sqlite"
    if target.exists():
        print(f"Olist fixture is ready: {sha256(target)}")
        return
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gen_olist_db.py")],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    if not target.exists():
        raise RuntimeError("Olist generator completed without creating data/olist.sqlite")
    print(f"Olist fixture generated: {sha256(target)}")


def main() -> None:
    # Compose mounts the demo database directory read-only in API/worker,
    # with a separate writable uploads volume below it. OCI cannot create
    # a missing child mountpoint inside an already read-only parent mount.
    # Prepare the empty mountpoint while migration holds the writable volume.
    (DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
    prepare_chinook()
    prepare_olist()


if __name__ == "__main__":
    main()
