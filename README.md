# Discovery and Mapping of Global Flood Plains

A professional AI/Geospatial project template for flood extent discovery, segmentation, and visualization from satellite imagery.

## Project Overview
Flood events cause severe economic and human losses globally. This repository provides a clean, modular baseline for building flood mapping models using geospatial imagery, classical ML baselines, and extensible deep learning hooks.

## Problem Statement
Given multi-band satellite data and reference flood labels, predict flood extent maps that can support:
- disaster response planning,
- risk assessment,
- climate adaptation strategy,
- urban and watershed management.

## Repository Structure

```text
flood-mapping/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   └── flood_mapping_demo.ipynb
├── src/
│   └── flood_mapping/
│       ├── __init__.py
│       ├── config.py
│       ├── data_ingestion.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── model.py
│       ├── inference.py
│       ├── evaluation.py
│       ├── visualization.py
│       └── pipeline.py
├── models/
├── utils/
│   └── run_pipeline.py
├── results/
├── docs/
│   └── pipeline.md
├── tests/
│   └── test_evaluation.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset Description
Use geospatial data sources such as:
- **Satellite imagery**: Sentinel-1 SAR, Sentinel-2 MSI, Landsat.
- **Flood labels**: hand-annotated masks, Copernicus EMS flood layers, or curated government datasets.
- **Auxiliary data**: DEM, land cover, river networks, precipitation, and administrative boundaries.

Expected examples are documented in `data/README.md`.

## Methodology
The baseline workflow is:
1. **Data ingestion** (`data_ingestion.py`) using Rasterio/GeoPandas.
2. **Preprocessing** (`preprocessing.py`) for normalization and resizing.
3. **Feature extraction** (`features.py`) for pixel-wise features and indices (e.g., NDWI).
4. **Model training** (`model.py`) with a RandomForest baseline.
5. **Prediction** (`inference.py`) to produce flood masks.
6. **Evaluation** (`evaluation.py`) with IoU, Accuracy, Precision, Recall.
7. **Visualization** (`visualization.py`) for maps and overlays.

## Model Architecture
### Baseline (implemented)
- **RandomForestClassifier** for robust tabularized pixel features.

### Portfolio extension (recommended)
- Replace/augment with **U-Net**, **DeepLabV3+**, or **SegFormer** in PyTorch/TensorFlow for high-resolution semantic segmentation.

## Installation

```bash
git clone https://github.com/<your-username>/Discovery-and-Mapping-of-global-floodplain.git
cd Discovery-and-Mapping-of-global-floodplain
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e src
```

## Usage
Run the baseline pipeline:

```bash
PYTHONPATH=src python utils/run_pipeline.py
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

## Results and Sample Outputs
The repository includes scripts to create:
- Flood detection maps (`plot_flood_detection_map`)
- Before vs after satellite comparisons (`plot_before_after`)
- Prediction overlays (`plot_prediction_overlay`)

Save generated artifacts in `results/` and include representative images for reports or portfolio pages.

## Evaluation Metrics
Implemented metrics for performance tracking:
- **IoU (Intersection over Union)**
- **Accuracy**
- **Precision**
- **Recall**

## Professional Elements Included
- Modular package in `src/flood_mapping/`
- Unit tests in `tests/`
- Notebook demo in `notebooks/`
- Documentation in `docs/`
- MIT license, requirements, and gitignore

## Future Improvements
- Add experiment tracking (MLflow/W&B)
- Add model registry and reproducible training configs
- Include temporal flood forecasting with sequence models
- Add CI pipeline (lint + test + docs build)
- Integrate cloud-native geospatial processing workflows
