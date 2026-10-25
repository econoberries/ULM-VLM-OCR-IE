#!/usr/bin/env bash
# Submit repeated OCRBench v2 English KIE runs on Helix.
# Usage: bash slurm/run_kie_repeats.sh 5 [experiment_name]
# One Slurm array runs the 4 configurations in round-robin order, serially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS=(vlm_qwen35 vlm_minicpm ocr_qwen35 ocr_qwen25)
INPUT="data/english_kie.json"
DATASET="data/OCRBench_v2"

fail() { echo "ERROR: $*" >&2; exit 1; }

# Called by Slurm for each array task. Users should not call this directly.
if [[ "${1:-}" == "--worker" ]]; then
    experiment="${2:?Missing experiment name}"
    task_id="${SLURM_ARRAY_TASK_ID:?Run this through sbatch}"
    model_index=$(( (task_id - 1) % ${#MODELS[@]} ))
    repeat_number=$(( (task_id - 1) / ${#MODELS[@]} + 1 ))
    model="${MODELS[$model_index]}"
    printf -v repeat_label '%02d' "$repeat_number"
    out_dir="$REPO_ROOT/results/repeated_runs/$experiment/$model/run_$repeat_label"
    mkdir -p "$out_dir"
    output="$out_dir/predictions.json"

    cd "$REPO_ROOT"
    export TOKENIZERS_PARALLELISM=false
    echo "Model: $model | Repeat: $repeat_number | GPU: ${CUDA_VISIBLE_DEVICES:-unknown}"
    echo "Predictions: $output"

    case "$model" in
        vlm_qwen35)
            uv run python src/vlm/qwen35/run.py \
                --input "$INPUT" --dataset-root "$DATASET" --output "$output" ;;
        vlm_minicpm)
            uv run python src/vlm/minicpm/run.py \
                --input_file "$INPUT" --image_folder "$DATASET" --output_file "$output" ;;
        ocr_qwen35)
            uv run python src/ocr_lm/run_ocr-lm_qwen35.py \
                --input "$INPUT" --dataset-root . --output "$output" ;;
        ocr_qwen25)
            uv run python src/ocr_lm/run_ocr-lm_qwen25Instruct.py \
                --input "$INPUT" --dataset-root . --output "$output" ;;
        *) fail "Unknown configuration: $model" ;;
    esac
    echo "Finished: $model repeat $repeat_number"
    exit 0
fi

if [[ $# -lt 1 || $# -gt 2 || ! "${1:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash slurm/run_kie_repeats.sh REPEATS [experiment_name]" >&2
    echo "Example: bash slurm/run_kie_repeats.sh 5 baseline_oct25" >&2
    exit 2
fi

repeats="$1"
experiment="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "Invalid experiment name"
command -v sbatch >/dev/null 2>&1 || fail "sbatch not found; submit from the Helix login node"

cd "$REPO_ROOT"
for file in \
    "$INPUT" \
    src/vlm/qwen35/run.py \
    src/vlm/minicpm/run.py \
    src/ocr_lm/run_ocr-lm_qwen35.py \
    src/ocr_lm/run_ocr-lm_qwen25Instruct.py; do
    [[ -f "$file" ]] || fail "Missing $file (the Qwen VLM runner must be merged from feature/qwen35-english-kie)"
done
[[ -d "$DATASET" ]] || fail "Missing extracted benchmark folder: $DATASET"
[[ -d data/ocr_output ]] || fail "Missing PaddleOCR text output folder: data/ocr_output"

result_dir="results/repeated_runs/$experiment"
[[ ! -e "$result_dir" ]] || fail "$result_dir already exists; use a new experiment name"
log_dir="$REPO_ROOT/$result_dir/slurm_logs"
mkdir -p "$log_dir"

total=$(( repeats * ${#MODELS[@]} ))
echo "Submitting $total runs ($repeats repetitions x ${#MODELS[@]} model configurations)."
echo "Runs will execute ONE AT A TIME on A100 80GB, using the same input dataset."
job_id="$(sbatch --parsable \
    --job-name=kie-repeats \
    --partition=gpu-single \
    --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --gres=gpu:A100:1,gpumem_per_gpu:80GB \
    --mem=64G --time=04:00:00 \
    --array="1-${total}%1" \
    --output="$log_dir/%x-%A_%a.out" \
    --error="$log_dir/%x-%A_%a.err" \
    "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")" --worker "$experiment")"
printf 'Slurm array job: %s\nResults: %s\nLogs: %s\n' "$job_id" "$result_dir" "$log_dir"
echo "Check status: squeue -j ${job_id%%;*}"
