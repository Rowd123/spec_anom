# Contrats des fichiers intermédiaires

Tous les tableaux sont des fichiers **Parquet** accompagnés de
`<nom>.manifest.json`. Le manifeste contient la version de schéma, la configuration
effective et son SHA-256, la provenance, les versions logicielles, l'ordre des
variables et leurs unités. Les écritures passent par un fichier temporaire. En
mode reprise, la configuration et la liste de variables sont vérifiées puis les
lignes sont dédupliquées par `(source_id, channel_id, window_id, segment_id)`.

## Segments (`artifact_type=segments`)

Une ligne représente un masque final. Les identifiants stables sont `source_id`,
`channel_id`, `window_id` et `segment_id`. `window_start`, `window_end`,
`time_start`, `time_end`, `frequency_min` et `frequency_max` relient le masque aux
axes physiques. `source_segment_ids`, `merge_count`, `predicted_iou` et
`stability_score` assurent l'audit de qualité. Les caractéristiques décrites par
`FEATURE_MEANING` sont les valeurs physiques **brutes**, avant log, imputation ou
normalisation de modèle. La politique actuelle est volontairement stricte : une
valeur NaN ou infinie arrête l'apprentissage ou l'inférence; aucune imputation
silencieuse n'est faite.

Avec `extract --save-arrays DIR`, un NPZ séparé par fenêtre conserve STFT complexe,
PSD, axes et masques. Les caractéristiques peuvent ainsi être recalculées sans
SAM. Une fenêtre valide sans segment figure dans les journaux/métadonnées de
fenêtre mais ne crée pas de fausse ligne segment.

## Scores et partitions

Les scores conservent les identifiants, temps, `model_id`, `preprocessing_id` et
`score`. La convention est `higher_is_more_anomalous:-score_samples`: le score
n'est pas une probabilité. Les partitions conservent `model_id`, `cluster_label`,
`is_noise` et `membership_strength`; le bruit HDBSCAN n'est pas automatiquement
une anomalie métier.

Les décisions SPOT conservent chaque score, le seuil utilisé et `is_anomaly`.
L'état pickle contient un SPOT par `(source_id, channel_id)`, les identifiants du
modèle/prétraitement et la dernière clé. Les observations sont triées par série,
fenêtre et segment; les segments simultanés gardent l'ordre stable. La politique
actuelle est un seuil fixe après calibration (pas de mise à jour avec les données
analysées), ce qui rend les batchs équivalents. Une calibration sans assez
d'excès échoue explicitement.

## Frontières et reprise

Le découpage d'entrée ne doit jamais être employé pour recréer séparément les
fenêtres : `prepare_analysis_windows` trie, élimine les timestamps dupliqués selon
la politique « premier », régularise les discontinuités, interpole seulement les
petites lacunes bornées puis construit les fenêtres et leur recouvrement. Une
extraction doit donc recevoir une série logique complète (ou un lecteur futur
doit conserver `window_size-1` points de halo). Les batchs `score-iforest` ne
changent que la taille des matrices d'inférence.

## GPU et mémoire

* STFT et SAM acceptent `cpu`, `cuda`, `cuda:N` ou `auto`; MSST est explicitement
  CPU. SAM est construit une fois par processus, mis en évaluation et appelé sans
  gradients par la session existante. `automatic.points_per_batch` est le vrai
  batch de prompts de l'API SAM.
* Isolation Forest/HDBSCAN acceptent `backend=cpu|gpu|auto`; GPU requiert cuML et
  CUDA. L'apprentissage HDBSCAN reste global : aucun fit par lot n'est effectué.
  `n_jobs` ne concerne que le backend sklearn. Le backend GPU peut différer
  numériquement et les labels HDBSCAN sont arbitraires.
* `score-iforest --batch-size` borne la matrice d'inférence. L'extraction actuelle
  traite une fenêtre à la fois et n'annonce donc pas de faux paramètre de batch
  image. Il n'existe pas encore de repli automatique sur OOM : réduire
  `points_per_batch` (SAM) ou `--batch-size` (scores). Les dépendances CUDA sont
  optionnelles et `auto` retombe sur CPU.

L'apprentissage global peut nécessiter un sous-échantillonnage, mais cette version
n'en effectue aucun implicitement. Il faut produire explicitement un fichier de
segments d'apprentissage échantillonné, en conservant la provenance.
