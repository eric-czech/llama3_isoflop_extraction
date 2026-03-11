# Llama 3 IsoFLOP Extraction

This repository contains a small extraction and reproduction workflow for IsoFLOP curve data from the Llama 3 paper, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).

The extracted data comes from **Figure 2, "Scaling law IsoFLOPs curves"** in the paper.

## Contents

- `extract_isoflops_points.py`: Parses the source SVG figure, extracts point coordinates, converts them into plot-space values, and renders a reproduction plot.
- `isoflops.svg`: Source figure asset used for extraction.
- `isoflops_points.csv`: Extracted point data.
- `isoflops_ground_truth.png`: Reference image from the paper figure.
- `isoflops_reproduction.png`: Recreated plot generated from the extracted data and fitted curves.

## Usage

Run the extraction and plot reproduction script with the project virtual environment:

```bash
./.venv/bin/python extract_isoflops_points.py
```

This writes:

- `isoflops_points.csv`
- `isoflops_reproduction.png`
