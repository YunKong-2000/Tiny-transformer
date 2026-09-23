#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/verify_offline_bundle.py
python3 - <<'PY'
import platform
import sys
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('This bundle targets Linux x86_64, not the local macOS environment.')
if sys.version_info < (3, 9):
    raise SystemExit('Python >= 3.9 is required; NGC 25.08 uses Python 3.12.')
import numpy, torch
print('Existing runtime:', torch.__version__, torch.version.cuda, numpy.__version__)
PY
# The local-JSON encode/decode API needs only the native tokenizers wheel.
# Its Hugging Face download helpers are unused. Keep NGC's other libraries intact.
python3 -m pip install --no-index --no-deps --find-links wheelhouse 'tokenizers==0.21.4'
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python3 - <<'PY'
import json
from pathlib import Path
from tiny_transformer.tokenizer import Tokenizer
info = json.loads(Path('BUNDLE.json').read_text())
tokenizer = Tokenizer.load(Path(info['data_path']) / 'tokenizer.json')
text = 'Once upon a time, a little bird found a friend.'
assert tokenizer.decode(tokenizer.encode(text)) == text
print('Offline tokenizer ready:', tokenizer.vocab_size, 'tokens')
PY
