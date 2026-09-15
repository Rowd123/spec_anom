# spectral-anomaly

Pipeline modulaire de segmentation et d'analyse atypique de séries temporelles :

```text
DataFrame → qualité/fenêtres → STFT ou MSST → image SAM
          → SAM automatique ou guidé → fusion → caractéristiques
          ├─ Isolation Forest → score d'atypicité → décision SPOT
          └─ HDBSCAN → identifiant de cluster
```

## Architecture

Le cœur public est découpé par responsabilité : `energy.py` prépare les fenêtres,
`spectral.py` fournit le contrat spectral commun, `segmentation.py` unifie les deux
modes SAM, `masks.py` traite les recouvrements, `features.py` recalcule les
caractéristiques finales, puis `preprocessing.py`, `models.py`, `spot.py` et
`pipeline.py` composent les deux branches ML. Les anciens modules `msst.py` et
`structure.py` restent accessibles pour les analyses historiques encore couvertes,
mais ne forment plus un pipeline concurrent.

Trois fichiers indépendants portent tous les hyperparamètres expérimentaux :

* `configs/spectral_analysis.json` : échantillonnage, qualité, fenêtres, STFT/MSST,
  PSD et device ;
* `configs/sam.json` : modèle/image, mode automatique ou `energy_contrast`, prompts
  et post-traitement ;
* `configs/models.json` : listes de variables indépendantes, split chronologique,
  Isolation Forest, SPOT, HDBSCAN et reproductibilité.

## STFT, MSST et sens physique

`analyze_spectrum` retourne toujours `SpectralResult`. `values` est la carte
sélectionnée pour la segmentation ; `stft` et `psd` sont conservées en parallèle.
Ainsi un masque MSST fournit légitimement sa géométrie/morphologie, mais
`integrated_energy`, `mean_energy_density`, `central_frequency`,
`frequency_dispersion` et `local_energy_contrast` sont évaluées sur la PSD STFT
alignée, jamais sur une pseudo-énergie MSST. Durée, largeur, aire et variations de
forme dépendent seulement du masque et des axes physiques. `FEATURE_MEANING` rend
cette distinction inspectable.

### Convention de tramage et axes

Le tramage est défini par le projet avant tout appel FFT, et non par NumPy,
Torch ou ssqueezepy. Avec `center=true`, les centres sont les échantillons
`0, hop_length, 2 hop_length, ...` strictement antérieurs à la fin du signal ;
la fenêtre est complétée à gauche et à droite selon `padtype`. Avec
`center=false`, seules les fenêtres entièrement contenues sont produites et le
temps publié est leur centre géométrique
`(start + (window_length - 1) / 2) / sampling_frequency`. CPU et GPU reçoivent
donc exactement les mêmes trames pondérées et retournent la même géométrie.
`center` ne retire jamais la moyenne : cette opération est contrôlée uniquement
par `signal_preprocessing.remove_mean`.

L'axe fréquentiel est `rfftfreq(n_fft, 1/fs)`. Augmenter `n_fft` densifie cette
grille par zero-padding mais n'améliore pas à lui seul la résolution physique,
qui dépend surtout de `window_length`, de la fenêtre et de la durée observée.

### Convention PSD

La STFT est le FFT brut des trames pondérées. Pour `scaling=density` et un signal
réel unilatéral, la PSD vaut `|X|² / (fs × sum(w²))` pour DC et Nyquist, et le
double pour les bins intérieurs positifs. Son unité est donc l'unité du signal
au carré par hertz et son intégrale fréquentielle respecte Parseval pour chaque
trame. Les seules options actuellement acceptées sont `density` et
`one_sided=true`; toute autre valeur provoque une erreur explicite.

## GPU sans faux accélérateur

Les devices spectraux acceptent `auto`, `cpu`, `cuda` ou `cuda:N`. La STFT CUDA
utilise réellement `torch.stft` et conserve les tenseurs sur GPU jusqu'au résultat ;
la MSST actuelle (réassignation sur la grille STFT, accélérée par Numba) est explicitement CPU et
`auto` retombe donc sur CPU. Demander CUDA pour MSST produit une erreur plutôt que
de simuler une accélération. SAM utilise le device du modèle PyTorch.

Les modèles acceptent `backend=auto|cpu|gpu`. CPU utilise scikit-learn ; GPU utilise
les implémentations cuML d'Isolation Forest et HDBSCAN. `auto` choisit cuML seulement
si cuML **et** CUDA sont disponibles, sinon CPU. SPOT reste sur CPU. Ces backends
partagent les contrats NumPy `score`/`cluster_id`. `cluster_id == -1` n'est pas une
décision d'anomalie ; `predicted_iou` et `stability_score` restent des métadonnées
SAM ; le score atypique et le booléen SPOT sont deux colonnes distinctes.

## Fusion auditable

Le post-traitement construit un graphe non orienté : deux masques sont reliés si
leur IoU dépasse `iou_threshold`, ou si le critère optionnel de containment
`|A∩B|/min(|A|,|B|)` dépasse son seuil. Chaque composante connexe est un groupe :
`A↔B` et `B↔C` produit donc `{A,B,C}`, indépendamment de l'ordre. `merge` calcule
l'union pixel à pixel ; `deduplicate` garde le plus grand support. Chaque résultat
conserve `source_segment_ids`, `merge_count` et les liens/IoU. Les caractéristiques
sont ensuite recalculées à partir du masque final, et ne sont jamais moyennées.

## Exemples

```bash
python examples/spectral_analysis_usage.py --config configs/spectral_analysis.json
python examples/sam_usage.py --spectral-config configs/spectral_analysis.json --sam-config configs/sam.json
python examples/sam_spectral_usage.py --spectral-config configs/spectral_analysis.json --sam-config configs/sam.json --output sam_spectral_diagnostic.html
python examples/pipeline_usage.py --spectral-config configs/spectral_analysis.json --sam-config configs/sam.json --models-config configs/models.json
```

`sam_usage.py` exécute réellement SAM 2 avec le checkpoint, la configuration de
modèle et le device déclarés dans `configs/sam.json`. Il montre le signal riche
(ton permanent, bouffée et chirp), la représentation, l'image RGB exacte, tous
les contours bruts sans addition d'identifiants, les masques individuels, leurs
métadonnées et le résultat du post-traitement. Le device effectivement choisi est
affiché. Le modèle/générateur est construit une fois dans une
`SAMSegmentationSession`, puis réutilisable pour toutes les fenêtres. Le diagnostic
synthétique de fusion, qui ne prétend pas évaluer SAM, est désormais isolé dans
`examples/mask_merge_diagnostic.py`.

La conversion en image applique `abs`, `log1p` (ou `linear`), les percentiles et
la mise à l'échelle uint8 **indépendamment pour chaque fenêtre**, puis répète le
gris sur trois canaux. Une fenêtre physiquement peu énergétique peut donc sembler
visuellement contrastée : cette image n'est pas une calibration énergétique.

Trois IoU distinctes ne doivent pas être confondues : `predicted_iou` est
l'estimation de qualité interne à SAM et non un recouvrement entre deux objets ;
`energy_contrast.mask_iou_threshold` déduplique les masques issus de prompts
guidés différents ; `mask_postprocessing.iou_threshold` relie ensuite les objets
dans le graphe de fusion. Les liens du graphe, leur IoU et leur containment sont
conservés avec les métadonnées complètes des segments SAM sources.
Le pipeline réel `dataframe_to_segments` accepte directement un DataFrame et ses
adaptateurs SAM peuvent également être injectés dans les tests ou traitements par
lots ; aucun CSV intermédiaire n'est requis.

SAM 2 demeure optionnel et aucun poids n'est téléchargé. Installer une version de
PyTorch adaptée puis SAM 2 depuis son dépôt officiel et renseigner le checkpoint.

### Référence STFT → SAM → caractéristiques

`sam_spectral_usage.py` force localement la STFT et SAM automatique, construit une
seule `SAMSegmentationSession`, puis affiche le signal, la STFT, l'image RGB exacte,
les contours réels des masques bruts, les contours retenus et leur table. Par
défaut la table décrit les masques SAM bruts ; `--postprocess` applique les options
de fusion de `sam.json` et décrit alors les objets finaux.

L'entrée de cet exemple est un `DataFrame` indexé par des timestamps UTC et muni
d'une colonne `quality`. Il contient des flags invalides et une petite lacune de
timestamp. `prepare_analysis_windows` applique les critères de qualité et
l'interpolation configurés avant la STFT. Le panneau temporel distingue le signal
préparé, les observations valides, les flags rejetés et les valeurs interpolées.
Les centres de trames STFT, les contours SAM et leurs infobulles sont replacés sur
l'axe absolu fourni par l'index, et non affichés sur un simple compteur d'échantillons.
`--window-id` permet de choisir la fenêtre préparée à examiner.

La table est volontairement limitée à `segment_id`, `time_frequency_area`,
`duration`, `frequency_width`, `central_frequency`, `frequency_dispersion`,
`integrated_spectral_power`, `mean_spectral_power_density`, `temporal_variation`,
`frequency_variation` et `local_energy_contrast`. L'image uint8 sert uniquement à
SAM : les quantités de puissance utilisent la PSD de la STFT complexe originale.
Aucune orientation, cohérence, linéarité, significance, PCA ou caractéristique de
tenseur de structure n'est calculée ou affichée par cet exemple.

L'exemple spectral contient deux signaux : deux tons connus pour contrôler les
fréquences, puis un ton permanent, une bouffée localisée et un chirp. Lorsque
`visualization.compare_representations=true`, il calcule directement STFT et
MSST avec `analyze_spectrum` sur les mêmes fenêtres et vérifie l'égalité exacte
de leurs axes avant de tracer les deux cartes. L'ancien exemple morphologique
`compare_stft_msst.py` reste séparé et ne participe pas à cette comparaison.

## Tests

```bash
python -m pytest
```
