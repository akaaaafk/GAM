

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


_DEFAULT_DATA = os.path.join(_PROJECT_ROOT, "data_cut")
_DEFAULT_OUTDIR = os.path.join(_PROJECT_ROOT, "results_qwen", "ruler_server")


def _ensure_server_defaults():
    argv = sys.argv[1:]
    has_data = any(arg == "--data" for arg in argv)
    has_outdir = any(arg == "--outdir" for arg in argv)
    if not has_data:
        argv = ["--data", _DEFAULT_DATA] + argv
    if not has_outdir:
        argv = ["--outdir", _DEFAULT_OUTDIR] + argv
    sys.argv = [sys.argv[0]] + argv


if __name__ == "__main__":
    _ensure_server_defaults()
    from eval_qwen.ruler_run import main
    main()
