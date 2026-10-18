"""Master test runner for BART-Survival test suite.
"""

import sys
import os
import importlib
from pathlib import Path

# Ensure repo root and tests directory are in sys.path
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

TEST_MODULES = [
    "test_simulation",
    "test_causal_pdp",
    "test_estimator",
    "test_dart_sparsity",
    "test_bart_survival",
]

def main():
    print("=" * 60)
    print("Running BART-Survival Comprehensive Test Suite")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0

    for mod_name in TEST_MODULES:
        print(f"\n[TEST SUITE] Running {mod_name}...")
        try:
            mod = importlib.import_module(mod_name)
            test_funcs = [getattr(mod, f) for f in dir(mod) if f.startswith("test_") and callable(getattr(mod, f))]
            for func in test_funcs:
                print(f"  -> {func.__name__} ...", end=" ", flush=True)
                func()
                print("PASSED")
                total_passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            total_failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Test Suite Summary: {total_passed} passed, {total_failed} failed.")
    print("=" * 60)
    if total_failed > 0:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

