# Analyse en lecture seule des artefacts

`analyze_results.py` inspecte les tables, manifestes, modèles pickle et fenêtres
NPZ produits en amont. Il n'entraîne aucun modèle et n'écrit que sous `--output`.
Chaque rapport inclut `input_inspection.json`, qui décrit les structures réellement
lues. L'absence d'une fenêtre NPZ n'empêche pas les rapports tabulaires : la galerie PDF
affiche alors un avertissement pour le segment concerné.

## Isolation Forest et SHAP

```bash
python analysis/analyze_results.py iforest \
  --segments artifacts/segments.parquet --scores artifacts/scores.parquet \
  --model artifacts/iforest.pkl --windows artifacts/windows --top-n 20 \
  --output analysis_output/iforest

python analysis/analyze_results.py iforest \
  --segments artifacts/segments.parquet --scores artifacts/scores.parquet \
  --model artifacts/iforest.pkl --windows artifacts/windows --top-n 20 \
  --shap --shap-samples 1000 --shap-background 50 --shap-evals 256 --seed 42 \
  --output analysis_output/shap
```

SHAP emploie `KernelExplainer` sur la fonction exacte
`-model.score_samples(saved_preprocessor.transform(raw_features))`. Installer
SHAP séparément avec `pip install shap` si nécessaire.

## HDBSCAN et analyse croisée

```bash
python analysis/analyze_results.py hdbscan \
  --segments artifacts/segments.parquet --clusters artifacts/clusters.parquet \
  --model artifacts/hdbscan.pkl --scores artifacts/scores.parquet \
  --windows artifacts/windows --medoid-sample 2000 \
  --output analysis_output/hdbscan

python analysis/analyze_results.py combined \
  --scores artifacts/scores.parquet --clusters artifacts/clusters.parquet \
  --output analysis_output/combined

python analysis/analyze_results.py all \
  --segments artifacts/segments.parquet --scores artifacts/scores.parquet \
  --model artifacts/iforest.pkl --clusters artifacts/clusters.parquet \
  --hdbscan-model artifacts/hdbscan.pkl --windows artifacts/windows \
  --output analysis_output/all
```

La projection PCA est la référence déterministe. `--umap` utilise UMAP si le
module est disponible (`pip install umap-learn`) et revient sinon à PCA. La
projection reste une visualisation et ne remplace jamais l'espace ajusté par
HDBSCAN. Le médoïde est toujours une observation réelle ; au-delà de
`--medoid-sample`, il est approximé sur un échantillon reproductible et cette
méthode est inscrite dans le rapport.

Sorties principales : tables CSV et rapports PDF des scores, clusters, représentants et
bruit/non-assignés ; galeries PDF avec PSD/STFT et contour du masque ; projection PDF
des clusters ; distributions croisées score/membership ; importances et
contributions SHAP lorsque demandées. Ni le score Isolation Forest ni la force
d'appartenance HDBSCAN ne sont interprétés comme des probabilités.

La génération PDF utilise Matplotlib. Si le module n'est pas présent, le script
indique explicitement de l'installer avec `pip install matplotlib`. Aucun fichier
HTML n'est généré.
