#!/usr/bin/env bash
#SBATCH --job-name=qwen2.5-14b_ocr_kie
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
uv run python scripts/run_ocr-lm_qwen25Instruct.py \
--input data/english_kie.json \
--dataset-root . \
--output results/ocr_qwen25-14b-INSTRUCT-kie.json


#SBATCH --partition=gpu-single
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1     # uppercase A100; lowercase a100 is rejected
#SBATCH --mem=64G
--gres=gpu:1 gives whichever GPU is free (my test job got an H200), so the timings wouldn't be comparable. Before submitting, check that Helix accepts the settings: sbatch --test-only <file>.sbatch.

