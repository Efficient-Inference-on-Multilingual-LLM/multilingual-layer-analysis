# Multi-lingual Layer Analysis
Official implementation for extracting LLM hidden states and computing topological clustering metrics (Silhouette Score, LAPE) for multilingual representation analysis.

## Overview

This framework allows for:
1.  **Layer-wise Extraction:** Extracting hidden states from specific transformer depths (e.g., Early, Middle, Deep layers) across diverse language families.
2.  **Topological Evaluation:** Computing **Silhouette Scores** and **Language Activation Probability Entrophy (LAPE)** to quantify how well model clusters align with linguistic taxonomy.

## Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

### Prerequisites
* **Python 3.12** (Managed automatically by `uv`)
* **GPU:** NVIDIA GPU with CUDA 12 support (Tested on NVIDIA L4).

### Install uv
If you do not have `uv` installed, install it via the standalone installer:
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

### Install Dependencies
```
uv sync
```

## How to Run

### Activation Extraction
Run this file: ⁠ scripts/run_nt.sh ⁠. Configuration can be adjusted inside the file.


### Silhouette Score

#### Calculate score matrices per layer
Run this command:
```
python -m src.main.run_silhouette
```


#### Calculate score, grouped by a certain category (e.g. script, family, etc.)
Run this command (change the configuration as needed):
```
python -m src.main.run_ss_perlayer --model_names <list-of-models> --residual-positions <extraction-point> --cuda-device <cuda-id>
```


### LAPE (Language-specific Neuron Calculation)

Run this file: ⁠ `scripts/run_all_lape.sh` ⁠. 
> Configuration can be adjusted inside the file.