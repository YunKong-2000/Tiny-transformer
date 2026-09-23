#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir="${1:-runs/smoke-$(date +%Y%m%d-%H%M%S)}"
python3 -m tiny_transformer.prepare --source smoke --train-docs 500 --val-docs 50 --output "$run_dir/data"
python3 -m unittest discover -s tests -v
python3 -m tiny_transformer.train --config configs/smoke.json --data "$run_dir/data" --output "$run_dir/train" --device cpu
python3 -m tiny_transformer.generate --checkpoint "$run_dir/train/last.pt" --device cpu --max-new-tokens 16
python3 -m tiny_transformer.benchmark --checkpoint "$run_dir/train/last.pt" --device cpu --prompt-length 16 --new-tokens 8 --warmup 1 --repeats 3 --output "$run_dir/benchmark.json"
python3 -m tiny_transformer.profile --checkpoint "$run_dir/train/last.pt" --device cpu --phase decode --seq-len 16 --output "$run_dir/profile"
