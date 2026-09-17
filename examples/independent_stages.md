# Exemple reproductible des étapes indépendantes

Installer l'I/O Parquet (`pip install -e '.[io,hdbscan]'`), puis :

```bash
# Extraction unique; SAM est chargé une seule fois dans ce processus.
spectral-anomaly extract data.csv artifacts/segments.parquet \
  --index-col time --value-col value --source-id machine-01 --channel-id vibration-x \
  --spectral-config configs/spectral_analysis.json --sam-config configs/sam.json \
  --save-arrays artifacts/windows --resume

# Nouveau processus, sans accès au CSV ni à SAM.
spectral-anomaly train-iforest artifacts/segments.parquet artifacts/iforest.pkl \
  --config configs/models.json
spectral-anomaly score-iforest artifacts/segments.parquet artifacts/iforest.pkl \
  artifacts/scores.parquet --batch-size 65536

# Branche de clustering indépendante.
spectral-anomaly fit-hdbscan artifacts/segments.parquet artifacts/hdbscan.pkl \
  artifacts/clusters.parquet --config configs/models.json
spectral-anomaly predict-hdbscan new-segments.parquet artifacts/hdbscan.pkl \
  artifacts/new-clusters.parquet

# SPOT ne lit que deux fichiers de scores et sauvegarde son état.
spectral-anomaly spot calibration-scores.parquet analysis-scores.parquet \
  artifacts/decisions.parquet artifacts/spot-state.pkl --config configs/spot.json
```

Pour comparer le coût des batchs sur exactement le même fichier :

```bash
for n in 256 4096 65536; do
  /usr/bin/time -v spectral-anomaly score-iforest artifacts/segments.parquet \
    artifacts/iforest.pkl "artifacts/scores-$n.parquet" --batch-size "$n"
done
```

CUDA étant asynchrone, un benchmark Python GPU doit appeler
`torch.cuda.synchronize()` immédiatement avant de démarrer et avant d'arrêter le
chronomètre. Validation GPU à exécuter sur une machine équipée :

```bash
python -c 'import torch, cuml; assert torch.cuda.is_available()'
pytest -q -m gpu
spectral-anomaly score-iforest artifacts/segments.parquet artifacts/iforest-gpu.pkl \
  artifacts/scores-gpu.parquet --batch-size 65536
```
