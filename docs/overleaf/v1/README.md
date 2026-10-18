# Overleaf manuscript — Version 1

**Date:** 18 October 2026  
**Project:** ULM-VLM-OCR-IE  
**Archive:** `ULM_VLM_OCR_IE_Overleaf_v1.zip`

Initial modular Overleaf/LaTeX manuscript for the document key information extraction benchmark project. The archive includes independently editable section files, tables, result figures, references, and selected document images used in qualitative error analysis.

**Qwen3.5-9B results included:** 400 English KIE examples from OCRBench v2 (FUNSD, POIE, Receipt, SROIE); official mean F1 **0.8699**, reference-format-corrected diagnostic mean F1 **0.8724**, mean inference latency **4.9181 s/document**, peak allocated CUDA memory up to **22.694 GiB**, and structured-output validity checks. The draft also documents a targeted generation-length sensitivity experiment and examples of benchmark annotation and output-format issues.

**Status:** Version 1 manuscript draft, not final submission. Cross-model comparisons and joint conclusions remain to be completed by the team.

**How to use:** Download the archive, upload it as a new project in Overleaf, and compile `main.tex` with pdfLaTeX. Keep later revisions as separate snapshots (e.g., `v2`) so Version 1 remains reproducible.
