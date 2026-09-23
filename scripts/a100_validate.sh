#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir="${1:-runs/a100-$(date +%Y%m%d-%H%M%S)}"
python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); print(torch.__version__, torch.version.cuda)'
python3 -m unittest discover -s tests -v
python3 -m tiny_transformer.prepare --source smoke --train-docs 500 --val-docs 50 --output "$run_dir/data"
for precision in fp32 bf16 fp16; do
  python3 -m tiny_transformer.train --config configs/smoke.json --data "$run_dir/data" --device cuda --precision "$precision" --steps 10 --output "$run_dir/$precision"
done
python3 -m tiny_transformer.check_ops --operator attention --backend sdpa --device cuda --precision bf16 --backward --output "$run_dir/sdpa-check.json"
python3 -m tiny_transformer.benchmark --checkpoint "$run_dir/bf16/last.pt" --device cuda --precision bf16 --op attention=sdpa --prompt-length 32 --new-tokens 16 --output "$run_dir/baseline.json"
TORCH_LOGS="graph_breaks,recompiles" python3 -m tiny_transformer.benchmark --checkpoint "$run_dir/bf16/last.pt" --device cuda --precision bf16 --op attention=sdpa --compile --prompt-length 32 --new-tokens 16 --output "$run_dir/compile.json" 2> "$run_dir/compile.log"
python3 -m tiny_transformer.profile --checkpoint "$run_dir/bf16/last.pt" --device cuda --precision bf16 --op attention=sdpa --phase decode --seq-len 32 --output "$run_dir/profile"
