"""Container probes. Never print service credentials or exception details."""
import os
import sys
import time
import urllib.request
from pathlib import Path


def main():
    try:
        if sys.argv[1] == "api":
            with urllib.request.urlopen("http://127.0.0.1:8000/health/ready", timeout=8) as response:
                return 0 if response.status == 200 else 1
        if sys.argv[1] == "worker":
            path = Path(os.getenv("WORKER_HEALTH_FILE", "/tmp/nexusmind-worker-health"))
            return 0 if 0 <= time.time() - float(path.read_text()) < 45 else 1
    except Exception:
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
