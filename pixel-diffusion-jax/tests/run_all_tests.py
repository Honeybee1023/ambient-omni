from __future__ import annotations

import importlib
import sys
from pathlib import Path
import traceback


TEST_MODULES = [
    "test_networks",
    "test_loss",
    "test_sampler",
    "test_training_step",
]


def main():
    tests_dir = Path(__file__).resolve().parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    failures = 0
    for module_name in TEST_MODULES:
        try:
            module = importlib.import_module(module_name)
            for name in dir(module):
                if name.startswith("test_") and callable(getattr(module, name)):
                    getattr(module, name)()
            print(f"PASS {module_name}")
        except Exception:
            failures += 1
            print(f"FAIL {module_name}")
            traceback.print_exc()
    raise SystemExit(failures)


if __name__ == "__main__":
    main()
