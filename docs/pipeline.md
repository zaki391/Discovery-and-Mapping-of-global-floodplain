# Flood Mapping Pipeline

The project follows a modular, production-style pipeline:

1. **Data ingestion** with Rasterio and GeoPandas.
2. **Preprocessing** using normalization and optional resizing.
3. **Feature extraction** from multi-band imagery.
4. **Model training** with RandomForest (baseline) and extension points for deep learning.
5. **Flood prediction** into scene-level masks.
6. **Evaluation** with IoU, Accuracy, Precision, Recall.
7. **Visualization** of maps, before/after imagery, and overlays.

The reference implementation lives under `src/flood_mapping/`.
