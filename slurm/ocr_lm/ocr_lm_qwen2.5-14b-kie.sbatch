#!/usr/bin/env bash
#SBATCH --job-name=qwen2.5-14b_ocr_kie
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
##SBATCH --gres=gpu:A100:1
#SBATCH --gres=gpu:A100:1,gpumem_per_gpu:80GB
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
#old: --output results/ocr_qwen25-14b-INSTRUCT-kie.json
uv run python src/ocr_lm/run_ocr-lm_qwen25Instruct.py \
--input data/english_kie.json \
--dataset-root . \
--output results/ocr_lm/ocr_qwen25-14b-INSTRUCT-kie.json
