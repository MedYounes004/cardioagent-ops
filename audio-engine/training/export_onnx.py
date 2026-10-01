"""Export the selected classifier to the serving model path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export(model_path: str | Path, output_path: str | Path) -> Path:
    import joblib

    model = joblib.load(model_path)
    converted = convert_sklearn(model, initial_types=[("features", FloatTensorType([None, 3]))])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(converted.SerializeToString())
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", default="models/cardio_model.onnx")
    args = parser.parse_args()
    export(args.model, args.output)


if __name__ == "__main__":
    main()
