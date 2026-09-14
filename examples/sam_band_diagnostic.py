"""Reproduce and diagnose the 45--47 s SAM 2 segmentation omission.

The input is the numeric Plotly payload from ``sam_automatic_masks.html`` (not a
screenshot), or an NPZ previously extracted by this script.  No spectral data
are recomputed, so prompts and automatic generation see exactly the same uint8
image and physical axes as the original run.
"""

from __future__ import annotations

import argparse
import base64
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from spectral_anomaly import (
    SAMAutomaticMaskSegmenter, SAMStructureSegmenter, pixel_to_time_frequency,
    time_frequency_to_pixel, validate_automatic_mask_options,
)

DEFAULT_CONFIG = Path(__file__).with_name("sam_band_diagnostic_config.json")


def _decode_array(value: Any) -> np.ndarray:
    """Decode ordinary or Plotly 6 binary-array JSON."""
    if isinstance(value, dict) and "bdata" in value:
        dtype = np.dtype(value["dtype"])
        result = np.frombuffer(base64.b64decode(value["bdata"]), dtype=dtype)
        if "shape" in value:
            shape = tuple(int(v) for v in str(value["shape"]).split(","))
            result = result.reshape(shape)
        return result
    return np.asarray(value)


def _plotly_data_from_html(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    marker = "Plotly.newPlot("
    decoder = json.JSONDecoder()
    # The last call is the user figure; earlier calls can belong to plotly.js.
    for position in reversed([i for i in range(len(text)) if text.startswith(marker, i)]):
        cursor = position + len(marker)
        try:
            _, used = decoder.raw_decode(text[cursor:])  # div id
            cursor += used
            cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
            if text[cursor] != ",":
                continue
            cursor += 1
            data, _ = decoder.raw_decode(text[cursor:].lstrip())
            if isinstance(data, list) and len(data) >= 2:
                return data
        except (json.JSONDecodeError, IndexError):
            continue
    raise ValueError(f"no usable Plotly.newPlot payload found in {path}")


def load_exact_input(html: Path | None, npz: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return magnitude, exact RGB SAM image, time, and frequency axes."""
    if npz is not None:
        with np.load(npz) as values:
            return tuple(values[name] for name in ("magnitude", "sam_image", "time", "frequency"))  # type: ignore[return-value]
    if html is None or not html.is_file():
        raise FileNotFoundError("the original HTML is absent; set input_html or input_npz")
    data = _plotly_data_from_html(html)
    original, supplied = data[0], data[1]
    log_magnitude = _decode_array(original["z"]).astype(float)
    grayscale = _decode_array(supplied["z"]).astype(np.uint8)
    time = _decode_array(supplied.get("x", original.get("x"))).astype(float)
    frequency = _decode_array(supplied.get("y", original.get("y"))).astype(float)
    if grayscale.shape != log_magnitude.shape or grayscale.shape != (len(frequency), len(time)):
        raise ValueError("the first two HTML heatmaps do not share compatible image axes")
    return np.expm1(log_magnitude), np.repeat(grayscale[..., None], 3, axis=2), time, frequency


@contextmanager
def automatic_filter_observer(generator: object, enabled: bool,
                              target_xy: tuple[float, float]) -> Iterator[dict[str, Any]]:
    """Temporarily observe AMG stages without editing the installed package."""
    report: dict[str, Any] = {"enabled": enabled, "raw_batches": [], "after_iou": [], "nms_inputs": []}
    if not enabled:
        yield report
        return
    import sam2.automatic_mask_generator as amg

    original_stability, original_nms = amg.calculate_stability_score, amg.batched_nms
    predictor = getattr(generator, "predictor", None)
    original_predict = getattr(predictor, "_predict", None)

    if original_predict is not None:
        def predict(*args, **kwargs):
            result = original_predict(*args, **kwargs)
            masks, scores = result[:2]
            report["raw_batches"].append(int(np.prod(tuple(masks.shape)[:2])))
            coordinates = kwargs.get("point_coords", args[0] if args else None)
            if coordinates is not None:
                coordinates = coordinates.detach().cpu().numpy() if hasattr(coordinates, "detach") else np.asarray(coordinates)
                coordinates = coordinates.reshape(len(coordinates), -1, 2)[:, 0]
                distances = np.linalg.norm(coordinates - np.asarray(target_xy), axis=1)
                nearest = int(np.argmin(distances))
                previous = report.get("nearest_grid_point")
                if previous is None or float(distances[nearest]) < previous["distance_px"]:
                    mask_values = masks[nearest].detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks[nearest])
                    score_values = scores[nearest].detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores[nearest])
                    y = int(np.clip(round(target_xy[1]), 0, mask_values.shape[-2] - 1))
                    x = int(np.clip(round(target_xy[0]), 0, mask_values.shape[-1] - 1))
                    report["nearest_grid_point"] = {
                        "xy": coordinates[nearest].astype(float).tolist(),
                        "distance_px": float(distances[nearest]),
                        "raw_proposals": [
                            {"score": float(score_values[i]), "area": int(np.count_nonzero(mask_values[i])),
                             "covers_diagnostic_point": bool(mask_values[i, y, x] > 0)}
                            for i in range(len(mask_values))
                        ],
                    }
            return result
        predictor._predict = predict

    def stability(masks, *args, **kwargs):
        report["after_iou"].append(int(len(masks)))
        scores = original_stability(masks, *args, **kwargs)
        threshold = float(getattr(generator, "stability_score_thresh", 0.0))
        report.setdefault("after_stability", []).append(int((scores >= threshold).sum()))
        return scores

    def nms(boxes, scores, *args, **kwargs):
        report["nms_inputs"].append(int(len(boxes)))
        keep = original_nms(boxes, scores, *args, **kwargs)
        report.setdefault("nms_outputs", []).append(int(len(keep)))
        return keep

    amg.calculate_stability_score, amg.batched_nms = stability, nms
    try:
        yield report
    finally:
        amg.calculate_stability_score, amg.batched_nms = original_stability, original_nms
        if original_predict is not None:
            predictor._predict = original_predict


def _mask_hover(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, "masque", "aucun masque")


def make_figure(magnitude, image, time_axis, frequency_axis, point, box, point_result,
                box_result, automatic_segments):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels: list[tuple[str, np.ndarray | None]] = [("Entrée exacte de SAM", None)]
    panels += [(f"Point M{i + 1} — score={score:.4f}", mask)
               for i, (mask, score) in enumerate(zip(point_result.masks, point_result.scores))]
    if box_result is not None:
        panels += [(f"Boîte M{i + 1} — score={score:.4f}", mask)
                   for i, (mask, score) in enumerate(zip(box_result.masks, box_result.scores))]
    panels += [(f"Automatique S{s.segment_id} — IoU={s.predicted_iou!s}, stabilité={s.stability_score!s}", s.mask)
               for s in automatic_segments]
    columns, rows = 3, (len(panels) + 2) // 3
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=[p[0] for p in panels])
    for index, (_, mask) in enumerate(panels):
        row, col = divmod(index, columns); row += 1; col += 1
        figure.add_trace(go.Heatmap(z=image[..., 0], x=time_axis, y=frequency_axis,
                                    colorscale="Gray", showscale=False, hovertemplate="t=%{x}<br>f=%{y}<br>intensité=%{z}<extra></extra>"), row, col)
        if mask is not None:
            figure.add_trace(go.Heatmap(z=mask.astype(np.uint8), customdata=_mask_hover(mask),
                                        x=time_axis, y=frequency_axis,
                                        colorscale=[[0, "rgba(0,0,0,0)"], [0.499, "rgba(0,0,0,0)"],
                                                    [0.5, "rgba(255,0,0,.45)"], [1, "rgba(255,0,0,.45)"]],
                                        showscale=False,
                                        hovertemplate="%{customdata}<br>t=%{x}<br>f=%{y}<extra></extra>"), row, col)
        figure.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers", name="point positif",
                                    marker={"color": "cyan", "size": 9, "symbol": "x"}, showlegend=index == 0), row, col)
        if box is not None:
            figure.add_shape(type="rect", x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                             line={"color": "yellow", "width": 2}, row=row, col=col)
    figure.update_layout(height=max(520, 330 * rows), title="Diagnostic SAM 2 — propositions individuelles", template="plotly_white")
    return figure


def main(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Relative paths intentionally follow the invocation directory, like the
    # original automatic-mask example and its configuration.
    resolve = lambda value: None if value is None else Path(value)
    magnitude, image, time_axis, frequency_axis = load_exact_input(resolve(config.get("input_html")), resolve(config.get("input_npz")))
    save_npz = resolve(config.get("save_extracted_npz"))
    if save_npz:
        np.savez_compressed(save_npz, magnitude=magnitude, sam_image=image, time=time_axis, frequency=frequency_axis)

    prompt = config["point"]
    point_xy = time_frequency_to_pixel(prompt["time"], prompt["frequency"], time_axis, frequency_axis)
    roundtrip = pixel_to_time_frequency(*point_xy, time_axis, frequency_axis)
    row, col = int(round(point_xy[1])), int(round(point_xy[0]))
    local = image[max(0, row-2):row+3, max(0, col-2):col+3, 0]
    print(f"Image exacte: {image.shape}, finite={np.isfinite(image).all()}, axes time/frequency: "
          f"{time_axis[0]}..{time_axis[-1]} / {frequency_axis[0]}..{frequency_axis[-1]}")
    print(f"Point physique=({prompt['time']}, {prompt['frequency']}), pixel={point_xy}, retour={roundtrip}, "
          f"intensité={image[row, col, 0]}, voisinage 5x5={local.min()}..{local.max()}")

    sam = config["sam"]
    checkpoint = resolve(sam["checkpoint"])
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint SAM 2 absent: {checkpoint}; extraction et axes vérifiés, inférence non exécutée")
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = "cuda" if sam["device"] == "auto" and torch.cuda.is_available() else ("cpu" if sam["device"] == "auto" else sam["device"])
    model = build_sam2(sam["model_config"], str(checkpoint), device=device)
    prompted = SAMStructureSegmenter(predictor=SAM2ImagePredictor(model), device=device)
    prompted.set_image(image)
    point_result = prompted.segment_point(point_xy, multimask_output=True)
    box_result, physical_box = None, None
    if config["box"]["enabled"]:
        b = config["box"]
        p0 = time_frequency_to_pixel(b["time_min"], b["frequency_min"], time_axis, frequency_axis)
        p1 = time_frequency_to_pixel(b["time_max"], b["frequency_max"], time_axis, frequency_axis)
        pixel_box = [min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])]
        physical_box = [b["time_min"], b["frequency_min"], b["time_max"], b["frequency_max"]]
        box_result = prompted.segment_box(pixel_box, multimask_output=True)
    options = validate_automatic_mask_options(config["automatic_mask_generation"])
    generator = SAM2AutomaticMaskGenerator(model, **options)
    auto = SAMAutomaticMaskSegmenter(generator=generator, device=device, **options)
    with automatic_filter_observer(generator, config.get("instrument_automatic_generator", True), point_xy) as stages:
        automatic = auto.generate_masks(image)
    print("Scores point:", point_result.scores.tolist())
    if box_result is not None: print("Scores boîte:", box_result.scores.tolist())
    print(f"Masques automatiques retenus: {len(automatic)}; instrumentation: {stages}")
    figure = make_figure(magnitude, image, time_axis, frequency_axis,
                         (prompt["time"], prompt["frequency"]), physical_box,
                         point_result, box_result, automatic)
    output = resolve(config["output_html"]); output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(f"Visualisation: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
