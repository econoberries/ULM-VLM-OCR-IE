#!/usr/bin/env bash
#SBATCH --job-name=qwen3.5-9b_ocr_kie
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
uv run python scripts/run_ocr-lm_qwen35.py \
--input data/english_kie.json \
--dataset-root . \
--output results/ocr_qwen35-9b-kie.json