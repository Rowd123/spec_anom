# Expérience par fenêtres temporelles

Cette expérience ajoute un extracteur déterministe, sans SAM, et réutilise les
implémentations existantes d'Isolation Forest, HDBSCAN et SPOT. L'expérience SAM
reste disponible avec ses configurations existantes.

## Étapes indépendantes

1. `prepare-windows` : CSV → fenêtres prétraitées en Parquet + audit de qualité.
2. `extract-windows` : fenêtres sauvegardées → caractéristiques en Parquet.
3. `split-windows` : caractéristiques → train, calibration, evaluation.
4. `train-iforest` / `score-iforest` : caractéristiques → modèle / scores.
5. `fit-hdbscan` / `predict-hdbscan` : caractéristiques → modèle / partition.
6. `spot` : scores de calibration + scores à analyser → décisions.

Changer les caractéristiques utilisées par un modèle ne demande de relancer que
son entraînement et les étapes aval. Changer les paramètres de l'extracteur exige
une nouvelle extraction, mais pas de relire le CSV. Changer la normalisation
exige de refaire le prétraitement.

Chaque Parquet est accompagné d'un `.manifest.json` : conserver les deux fichiers.
Les modèles mémorisent l'ordre exact des caractéristiques utilisées. Une signature
vérifie les paramètres d'extraction et de normalisation à l'inférence.

## Configuration du prétraitement

`configs/window_preprocessing.json` : fréquence, période, taille/recouvrement,
qualité, exclusions et normalisation. Les valeurs `mu` et `sigma` sont
intentionnellement `null` dans l'exemple : **renseigner vos constantes**, sinon la
commande échoue. Elles ne sont jamais estimées. Exemple de structure :

```json
"normalization": {
  "V1.MAG": {"mu": null, "sigma": null},
  "FREQ": {"mu": null, "sigma": null}
}
```

Les canaux correspondent à `--channel-id`; exécuter le prétraitement séparément
pour chaque série. Le calcul est `(x - mu) / sigma`, avec sigma strictement positif.
Le code appelle le même `prepare_analysis_windows` que l'expérience SAM : mêmes
règles de tri, doublons, grille temporelle, qualité, interpolation et exclusions.
Il utilise son signal, sans soustraction supplémentaire de la moyenne locale.
L'audit est sauvegardé dans `<windows>.quality.parquet`.

L'exemple propose 50 Hz et 3000 échantillons (60 s), avec un pas de 30 s : ce sont
des paramètres d'exemple, à adapter aux données. Les fenêtres incomplètes en fin
de série ne sont pas utilisées, conformément au prétraitement existant.

## Configuration de l'extraction

`configs/window_features.json` contrôle tous les descripteurs :

| Paramètre | Effet |
|---|---|
| `statistics` | Sous-ensemble de minimum, maximum, mean, variance, median, iqr, skewness, kurtosis |
| `variance_ddof` | Correction de variance, par défaut 0 |
| `moment_bias` | Option SciPy des estimateurs de skewness/kurtosis ; kurtosis est l'excès de Fisher |
| `variance_epsilon` | Seuil de variance pour considérer une fenêtre constante |
| `undefined_policy` | `zero` ou `error` pour moments/autocorrélations indéfinis |
| `differences` | Liste facultative : mean_absolute, maximum_absolute, variance ; vide = désactivé |
| `spectral.enabled` | Active les puissances par bandes |
| `spectral.bands_hz` | Bornes non uniformes, ordonnées et non chevauchantes ; couverture partielle autorisée |
| `spectral.window`, `detrend`, `nfft` | Paramètres du périodogramme ; `detrend: false` ne retire pas de moyenne |
| `spectral.top_changes` | Nombre des variations de bandes de plus grande valeur absolue ; 0 désactive |
| `temporal.lags_seconds` | Retards exacts en multiples de la période d'échantillonnage |
| `temporal.changes` | Ajoute les écarts d'autocorrélation par rapport à l'historique |
| `history.size`, `min_windows` | Taille maximale et minimum requis de l'historique |
| `history.warmup` | `drop`, `zero` ou `error` avant d'avoir assez d'historique |
| `history.reset_after_gap_seconds` | Écart maximal entre débuts successifs ; null = 1,5 fois le pas des fenêtres |
| `log_every` | Fréquence des messages de progression |

Les noms sont préfixés `wf_`. `wf_band_0_power` correspond à la première bande,
`wf_acf_0` au premier retard. Les variations utilisent le suffixe `_change`.
`wf_spectral_change_0` est la plus forte variation **signée** de puissance ;
`wf_spectral_change_0_frequency` est le centre de sa bande, pas la fréquence d'un
pic estimé. Les égalités sont départagées par l'ordre des bandes.

Le calcul spectral est un périodogramme unilatéral en densité, puis la somme des
bins sélectionnés multipliée par leur espacement. Les bandes sont semi-ouvertes
[bas, haut[, sauf la dernière borne si elle égale Nyquist. Elles doivent contenir
des bins et leur largeur doit atteindre au moins fs/N ; le zero-padding ne remplace
pas une fenêtre longue. Ces puissances sont celles du signal normalisé et non une
énergie physique en joules. Aucun estimateur de phase n'est utilisé.

L'autocorrélation au retard k est `sum(y[:-k]*y[k:]) / sum(y*y)` avec
`y = x - mean(x)`. Ce centrage définit la caractéristique, sans modifier les données
prétraitées. `wf_constant` indique une fenêtre où les moments normalisés sont
indéfinis et remplacés par zéro selon la politique choisie. Les différences sont
des différences finies non divisées par la période, pas des dérivées continues.

Les références historiques sont les médianes des fenêtres **strictement
précédentes**, séparément par source et canal. Elles sont réinitialisées après un
écart temporel trop grand. `wf_history_ready` indique la disponibilité de la référence.
Avec `warmup: drop`, les premières fenêtres alimentent la référence mais sont
absentes de la table finale. Une extraction se fait sur un historique complet :
ne pas traiter chaque fenêtre dans une invocation distincte.

Le prétraitement historique conserve son interpolation bornée, qui peut utiliser
un échantillon suivant pour remplir une petite lacune. La causalité annoncée ici
concerne la référence des caractéristiques, pas une nouvelle garantie de streaming
strict du prétraitement existant.

## Sélection indépendante pour chaque modèle

Dans `configs/window_models.json`, les listes `atypicality_features` et
`hdbscan_features` sélectionnent les colonnes sauvegardées à utiliser. Les champs
`atypicality_exclude` et `hdbscan_exclude` permettent de les retirer sans modifier
l'extraction, avec des motifs glob :

```json
"atypicality_exclude": ["wf_acf_*", "wf_kurtosis"],
"hdbscan_exclude": ["wf_spectral_change_*"]
```

Les noms inconnus et motifs ne correspondant à aucune caractéristique provoquent
une erreur. Une sélection vide est interdite. Pour utiliser les différences, les
activer à l'extraction puis ajouter les colonnes `wf_diff_*` souhaitées aux listes.
Les indicateurs `wf_constant`/`wf_history_ready` peuvent aussi être sélectionnés ou
ignorés comme les autres caractéristiques.

`preprocessing.scaling` concerne **la matrice de caractéristiques** avant modèle,
pas le signal : `none` (exemple), `robust` ou `standard`. Les deux derniers sont
ajustés sur les données de fitting uniquement. `log1p` est facultatif ; ne pas
l'appliquer à une caractéristique pouvant être inférieure ou égale à -1.
Aucune mise à l'échelle n'est choisie automatiquement pour cette expérience.

Les hyperparamètres des modèles, backends CPU/GPU existants, graine et fractions
sont ceux du fichier de modèles. L'extraction statistique utilise NumPy/SciPy CPU.
Le batch de scoring reste réglable par `--batch-size`.

## Commandes

Depuis la racine, installer `python -m pip install -e '.[io,hdbscan,test]'`.
Après avoir renseigné les constantes et adapté les paramètres :

```bash
spectral-anomaly prepare-windows donnees.csv artifacts/windows.parquet \
  --config configs/window_preprocessing.json \
  --index-col DATE_MESURE --value-col VIGY_1:VA_V1:MAG \
  --source-id VIGY_1 --channel-id V1.MAG --sep ';'

spectral-anomaly extract-windows artifacts/windows.parquet artifacts/window_features.parquet \
  --config configs/window_features.json

spectral-anomaly split-windows artifacts/window_features.parquet artifacts/window_splits \
  --config configs/window_models.json

spectral-anomaly train-iforest artifacts/window_splits/train.parquet artifacts/window_iforest.pkl \
  --config configs/window_models.json

spectral-anomaly score-iforest artifacts/window_splits/calibration.parquet artifacts/window_iforest.pkl \
  artifacts/window_calibration_scores.parquet --batch-size 65536

spectral-anomaly score-iforest artifacts/window_splits/evaluation.parquet artifacts/window_iforest.pkl \
  artifacts/window_evaluation_scores.parquet --batch-size 65536

spectral-anomaly spot artifacts/window_calibration_scores.parquet artifacts/window_evaluation_scores.parquet \
  artifacts/window_anomalies.parquet artifacts/window_spot.pkl --config configs/spot.json

spectral-anomaly fit-hdbscan artifacts/window_splits/train.parquet artifacts/window_hdbscan.pkl \
  artifacts/window_train_clusters.parquet --config configs/window_models.json

spectral-anomaly predict-hdbscan artifacts/window_splits/evaluation.parquet artifacts/window_hdbscan.pkl \
  artifacts/window_evaluation_clusters.parquet
```

`--datetime-format` vaut `ISO8601` par défaut ; fournir le format des dates si
nécessaire. Adapter `--sep` au CSV.

Les partitions sont chronologiques. Les fenêtres de la partition antérieure qui
partagent des échantillons avec la suivante sont supprimées. Une partition train
sauvegardée est utilisée en entier par `train-iforest` (pas de seconde découpe).
Les caractéristiques des partitions suivantes peuvent consulter l'historique
antérieur, comme en surveillance en ligne, jamais les fenêtres futures.

Le modèle HDBSCAN est ajusté au fichier fourni : fournir `train.parquet` pour une
évaluation séparée. On peut aussi partitionner toute la table pour une étude
exploratoire transductive, sans interpréter cela comme une évaluation hors
entraînement. `predict-hdbscan` conserve le chemin CPU existant avec le paquet
hdbscan et `prediction_data: true`.

SPOT conserve exactement le fonctionnement du dépôt : calibration POT et seuil
fixe. Il faut assez d'excès dans la partition de calibration pour les paramètres
choisis ; aucun changement silencieux de seuil ou de paramètres n'est effectué.

## Contrats et vérification

### Retracer le signal avec les fenêtres détectées (Plotly)

Après `spot`, tracer le CSV d'origine et les décisions sauvegardées :

```bash
spectral-anomaly plot-spot measurements.csv artifacts/window_anomalies.parquet \
  artifacts/window_spot.html \
  --index-col DATE_MESURE --value-col VIGY_1:VA_V1:MAG \
  --source-id VIGY_1 --channel-id V1.MAG --sep ';'
```

Adapter le CSV, le séparateur et les identifiants à ceux de `prepare-windows`.
`--datetime-format` fonctionne comme dans cette commande. Ouvrir le fichier HTML
dans un navigateur : Plotly est inclus, aucune connexion Internet n'est nécessaire.
Zoom, déplacement, légende et téléchargement SVG sont disponibles.

Le panneau supérieur affiche le signal brut dans ses unités d'origine, avec un
fond rouge sur toute la durée des fenêtres détectées. Les zones rouges qui se
chevauchent sont fusionnées pour garder une opacité constante : une seule fenêtre
détectée suffit à colorer la zone. Une zone sans rouge n'est pas nécessairement
normale : elle peut n'avoir aucune fenêtre évaluée (calibration, qualité, etc.).
Les bornes viennent de `window_start` et `window_end`, dernier échantillon inclus.
Le panneau inférieur montre les scores IF, le seuil SPOT et les alarmes, placés
en fin de fenêtre ; les infobulles donnent l'identifiant et les bornes de chaque
fenêtre. Les scores ne sont pas des probabilités. Le tracé ne réentraîne aucun
modèle et ne modifie aucune décision. SPOT reste la calibration POT à seuil fixe
déjà utilisée dans ce dépôt.

`prepared_windows` et `window_features` sont des types d'artefacts distincts des
segments SAM. Pour réutiliser la traçabilité des modèles, une fenêtre porte
`segment_id=0`, `observation_kind=window`, des bornes temporelles couvrant la fenêtre
et des bornes fréquentielles 0/Nyquist. Cela ne désigne pas un masque SAM.

Vérification ciblée :

```bash
python -m pytest tests/test_window_features.py tests/test_persistent_workflows.py \
  tests/test_pipeline_architecture.py tests/test_atypicality_pipeline.py
```

Les tests couvrent la normalisation fixe, le prétraitement partagé, la puissance
connue d'une sinusoïde, l'autocorrélation, les fenêtres constantes, l'historique
causal, les interruptions, la purge des chevauchements, la sélection indépendante,
les sauvegardes Parquet et l'intégration IF/HDBSCAN/SPOT sur données synthétiques.
