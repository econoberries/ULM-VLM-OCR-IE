# OCRBench v2 English KIE

We are a student group comparing two ways to do English key information extraction (KIE):

- End-to-end vision-language models (VLMs).
- OCR followed by a text-only language model (LM).

## Project folders

- `src/vlm/minicpm/`: MiniCPM-V-4.6
- `src/vlm/qwen35/`: Qwen3.5-9B
- `src/ocr_lm/`: OCR-to-LM pipeline
- `src/evaluation/`: shared data and evaluation utilities

## Setup

Clone the repository and install [uv](https://docs.astral.sh/uv/getting-started/installation/). From the project folder, run:

```sh
uv sync --locked
uv run python <script.py>
```

To add a shared package, run `uv add <package>` and commit both `pyproject.toml` and `uv.lock`.

## Data and evaluation

The OCRBench v2 data are downloaded separately and are not committed here. The [OCRBench v2 project page](https://99franklin.github.io/ocrbench_v2/) links to the code, dataset, and Hugging Face resources. We use the [official evaluation scripts](https://github.com/Yuliang-Liu/MultimodalOCR/tree/4e363b3/OCRBench_v2) without modifying them.
