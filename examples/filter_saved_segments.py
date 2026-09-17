"""Filter a saved feature table with configured study exclusions, without extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from spectral_anomaly import exclusion_signature, filter_excluded_segments, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spectral-config", required=True, type=Path)
    parser.add_argument("--source-column", default="source_id")
    args = parser.parse_args(argv)
    config = load_config(args.spectral_config, "spectral")
    frame = pd.read_parquet(args.input)
    filtered = filter_excluded_segments(
        frame, config["exclusions"], source_column=args.source_column
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(args.output, index=False)
    input_manifest = args.input.with_suffix(args.input.suffix + ".manifest.json")
    manifest = json.loads(input_manifest.read_text()) if input_manifest.exists() else {}
    manifest.update({
        "exclusions": config["exclusions"],
        "exclusion_signature": exclusion_signature(config["exclusions"]),
        "source_artifact": str(args.input),
        "rows_before_exclusion": len(frame),
        "rows_after_exclusion": len(filtered),
    })
    output_manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
    output_manifest.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"excluded={len(frame) - len(filtered)} retained={len(filtered)} output={args.output}")


if __name__ == "__main__":
    main()
