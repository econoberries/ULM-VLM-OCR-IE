# MiniCPM-V-4.6

Runs `openbmb/MiniCPM-V-4.6` on OCRBench v2 English **Key Information Extraction (KIE)** records.

## Files

* `run_one_record.py` — quick check on one KIE record
* `run.py` — run all English KIE records
* `minicpm_utils.py` — model loading and inference
* `ocrbench_utils.py` — OCRBench file handling
* `efficiency.py` — runtime and GPU-memory metrics

## Run

Quick check:

```bash
uv run python src/vlm/minicpm/run_one_record.py
```

Full cluster run:

```bash
sbatch slurm/vlm/minicpm_kie.sbatch
```

The full run uses only `"key information extraction en"` records and can resume from completed predictions.

