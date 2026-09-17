# Représentations spectrales et exclusions

`representation` accepte `stft` (défaut), `ssq_stft` et `msst`.

* `stft` emploie le tramage explicite du projet et NumPy ou `torch.fft`.
* `ssq_stft` appelle réellement `ssqueezepy.ssq_stft`. Il s'agit d'une SST
  d'ordre un, et elle n'est jamais appelée « MSST ». `ssqueezepy` 0.6.6 accepte
  les signaux 2-D : `ssq_stft.batch_size` borne donc le nombre de fenêtres dans
  un appel spectral. Ce batch est indépendant de `automatic.points_per_batch`
  de SAM.
* `msst` est l'extension multi-itérative déjà fournie par le projet : la STFT et
  sa dérivée sont calculées une seule fois avec les primitives `stft` et
  `phase_stft` de `ssqueezepy`; la carte de réallocation est composée le nombre
  de fois demandé, puis les coefficients STFT originaux sont réalloués une seule
  fois. Elle suit le principe du *Multisynchrosqueezing Transform* de Yu et al.,
  IEEE Transactions on Industrial Electronics 66(7), 2019,
  DOI `10.1109/TIE.2018.2868296`. Elle ne réapplique jamais `ssq_stft` à une SST.

La MSST multi-itérative reste CPU dans cette version, car `ssqueezepy` n'expose
pas cette transformation. Une demande CUDA explicite échoue au lieu de prétendre
accélérer le calcul. Pour `ssq_stft`, CUDA utilise le backend documenté de
`ssqueezepy` (`SSQ_GPU=1`) et nécessite PyTorch **et CuPy**. `auto` retombe sur
CPU; `cuda` indisponible échoue clairement. `dtype` sélectionne `float32` ou
`float64`. La mémoire CUDA et la durée de chaque batch sont journalisées.

Les paramètres `window`, `window_length`, `hop_length`, `n_fft`, `center`,
`padtype`, `dtype`, `frequency_min` et `frequency_max` sont communs. SST et MSST
exigent actuellement `center=true`, conformément au tramage de `ssqueezepy`.
`gamma`, `squeezing`, le batch SST et `msst.iteration_count` sont séparés.

`SpectralResult.values` alimente l'image SAM et donc la géométrie des masques.
Les caractéristiques géométriques utilisent ces masques et les axes secondes/Hz.
Les caractéristiques d'énergie restent calculées sur la PSD physique de la STFT
alignée (`signal²/Hz`), y compris en SST/MSST : l'énergie d'une SST/MSST réallouée
n'est pas assimilée silencieusement à une PSD.

## Exclusions

Chaque événement peut cibler `event_id`, un intervalle fermé `start`/`end`, une
`source_id`, ou une combinaison. Toutes les fenêtres dont le support contient au
moins un échantillon exclu sont rejetées avec `excluded_event`; les points exclus
ne sont pas interpolés en une continuité artificielle. Ces fenêtres n'atteignent
donc ni SAM, ni les statistiques, ni l'apprentissage/calibration. Les identifiants
inconnus sont journalisés ou refusés selon `unknown_id_policy`.

Les tables de segments portent `spectral_representation`, `spectral_signature`,
`exclusion_signature`, `source_id` et `window_end`. L'entraînement refuse un
mélange de signatures et le modèle sauvegardé refuse ensuite des données de score
incompatibles.

## Commandes

Extraction et entraînement complet avec SST et exclusions :

```bash
python examples/train_atypicality_usage.py \
  --input measurements.csv --time-col time --value-col value \
  --source-id machine-01 \
  --spectral-config configs/spectral_ssq_exclusions.json \
  --sam-config configs/sam.json --models-config configs/models.json \
  --scores-output artifacts/ssq_scores.csv
```

Pour MSST, copier la configuration d'exemple, fixer `"representation": "msst"`
et régler `msst.iteration_count`. Pour filtrer un Parquet de caractéristiques
existant sans relancer SAM :

```bash
python examples/filter_saved_segments.py \
  --input artifacts/segments.parquet \
  --output artifacts/segments-excluded.parquet \
  --spectral-config configs/spectral_ssq_exclusions.json
```
