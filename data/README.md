# Data Directory

This folder stores geospatial flood mapping datasets.

## Example layout

```text
data/
├── raw/
│   ├── example_satellite.tif
│   ├── example_flood_labels.tif
│   └── admin_boundaries.geojson
└── processed/
    ├── train_features.parquet
    └── val_features.parquet
```

> `raw/` contains immutable source data.
> `processed/` contains generated features and transformed rasters.
