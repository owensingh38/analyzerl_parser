"""Optional xG scoring helpers for parsed AnalyzeRL PBP data."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import polars as pl

XG_LABEL = "standard"
REQUIRED_ARTIFACT_KEYS = {
    "model",
    "preprocessor",
    "numeric_cols",
    "categorical_cols",
}


def _resolve_xg_helpers():
    from . import rl_xg

    return rl_xg


def _resolve_model_file(xg_model_path: str | PathLike[str]) -> Path:
    from .parse import _path

    path = _path(xg_model_path)
    if path.is_file():
        return path

    if path.is_dir():
        candidates = [
            path / f"xg_model_{XG_LABEL}.joblib",
            path / XG_LABEL / f"xg_model_{XG_LABEL}.joblib",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Could not find an xG model artifact. Expected a .joblib file, "
        f"{path / f'xg_model_{XG_LABEL}.joblib'}, or "
        f"{path / XG_LABEL / f'xg_model_{XG_LABEL}.joblib'}."
    )


def load_xg_artifact(xg_model_path: str | PathLike[str]) -> dict[str, Any]:
    """Load and validate a saved AnalyzeRL xG model artifact.

    Args:
        xg_model_path: Path to an ``xg_model_standard.joblib`` file or a folder
            containing that file.

    Returns:
        The loaded model artifact.

    Raises:
        FileNotFoundError: If no supported model file is found.
        ImportError: If ``joblib`` is unavailable.
        ValueError: If the loaded artifact is missing required keys.
    """
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("xG scoring requires joblib to load model artifacts") from exc

    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError:  # pragma: no cover - sklearn is optional at import time
        InconsistentVersionWarning = UserWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*If you are loading a serialized model.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            category=InconsistentVersionWarning,
        )
        artifact = joblib.load(_resolve_model_file(xg_model_path))
    if not isinstance(artifact, dict):
        raise ValueError("xG model artifact must be a dict saved by analyzerl_xg")

    missing = sorted(REQUIRED_ARTIFACT_KEYS - set(artifact))
    if missing:
        raise ValueError(f"xG model artifact is missing required keys: {missing}")

    if not artifact["numeric_cols"] and not artifact["categorical_cols"]:
        raise ValueError("xG model artifact does not define any feature columns")

    return artifact


def xg_model_source_columns() -> list[str]:
    """Return the PBP columns needed to build xG model features."""
    rl_xg = _resolve_xg_helpers()
    return list(dict.fromkeys(rl_xg.model_pbp_columns()))


def analyzerl_xg(*args: Any, **kwargs: Any) -> Any:
    """Build an AnalyzeRL xG model with the package-local xG implementation."""
    rl_xg = _resolve_xg_helpers()
    return rl_xg.analyzerl_xg(*args, **kwargs)


def _ensure_model_source_columns(rows: pl.DataFrame, rl_xg) -> pl.DataFrame:
    columns = rl_xg.model_pbp_columns()
    additions = [
        pl.lit(None).alias(column)
        for column in columns
        if column not in rows.columns
    ]
    if additions:
        rows = rows.with_columns(additions)

    return rows


def _add_missing_feature_columns(
    model_df: pl.DataFrame,
    shots: pl.DataFrame,
    artifact: dict[str, Any],
) -> pl.DataFrame:
    additions = []
    for column in artifact["numeric_cols"]:
        if column in model_df.columns:
            continue
        if column in shots.columns:
            series = shots.get_column(column)
            if series.dtype == pl.Boolean:
                additions.append(
                    series.cast(pl.Int8, strict=False)
                    .cast(pl.Float32, strict=False)
                    .alias(column)
                )
            else:
                additions.append(series.cast(pl.Float32, strict=False).alias(column))
        else:
            additions.append(pl.lit(None, dtype=pl.Float32).alias(column))

    for column in artifact["categorical_cols"]:
        if column in model_df.columns:
            continue
        if column in shots.columns:
            additions.append(shots.get_column(column).cast(pl.Utf8, strict=False).alias(column))
        else:
            additions.append(pl.lit(None, dtype=pl.Utf8).alias(column))

    return model_df.with_columns(additions) if additions else model_df


def _predict_xg(artifact: dict[str, Any], matrix, rl_xg) -> np.ndarray:
    model = artifact["model"]

    if hasattr(model, "predict_proba"):
        predictions = model.predict_proba(matrix)[:, 1]
    else:
        predictions = rl_xg.predict_scores(model, matrix)

    calibrator = artifact.get("calibrator")
    if calibrator is not None:
        predictions = calibrator.transform(predictions)

    return np.asarray(predictions, dtype=np.float64)


def apply_xg_to_pbp(
    rows: pl.DataFrame | pl.LazyFrame,
    xg_model_path: str | PathLike[str] | None,
) -> pl.DataFrame | pl.LazyFrame:
    """Apply a saved xG model to parsed PBP or frame rows.

    Args:
        rows: Parsed rows containing AnalyzeRL event columns.
        xg_model_path: Path to a saved xG model artifact or folder. ``None``
            returns ``rows`` unchanged.

    Returns:
        A Polars DataFrame with an ``xG`` column, or the original LazyFrame
        unchanged when ``xg_model_path`` is ``None``.
    """
    if xg_model_path is None:
        return rows

    rl_xg = _resolve_xg_helpers()
    artifact = load_xg_artifact(xg_model_path)
    frame = rows.collect() if isinstance(rows, pl.LazyFrame) else rows
    original_columns = [column for column in frame.columns if column != "xG"]
    output_columns = [*original_columns, "xG"]

    if frame.is_empty():
        return (
            frame.drop("xG") if "xG" in frame.columns else frame
        ).with_columns(pl.lit(None, dtype=pl.Float64).alias("xG"))

    frame = _ensure_model_source_columns(frame, rl_xg)
    frame = frame.with_row_index("_xg_row")
    shots = rl_xg.prepare_model_shots(frame.lazy()).collect()

    if shots.is_empty():
        return (
            frame.drop("_xg_row")
            .with_columns(pl.lit(None, dtype=pl.Float64).alias("xG"))
            .select(output_columns)
        )

    model_df = rl_xg.build_segment_features(shots)
    model_df = _add_missing_feature_columns(model_df, shots, artifact)
    matrix = rl_xg.make_sparse_matrix(
        model_df,
        artifact["numeric_cols"],
        artifact["categorical_cols"],
        preprocessor=artifact["preprocessor"],
    )
    predictions = _predict_xg(artifact, matrix, rl_xg)

    prediction_frame = shots.select("_xg_row").with_columns(
        pl.Series("xG", predictions)
    )

    if "xG" in frame.columns:
        frame = frame.drop("xG")

    return (
        frame.join(prediction_frame, on="_xg_row", how="left")
        .sort("_xg_row")
        .drop("_xg_row")
        .select(output_columns)
    )


def apply_xg_to_file(
    path: str | PathLike[str],
    xg_model_path: str | PathLike[str] | None,
    export_format: str,
) -> None:
    """Apply xG scoring to a CSV or Parquet PBP/frame export in place."""
    if xg_model_path is None:
        return

    path = Path(path)
    export_format = str(export_format).lower()

    if export_format == "parquet":
        rows = pl.read_parquet(path)
        scored = apply_xg_to_pbp(rows, xg_model_path)
        scored.write_parquet(path)
        return

    if export_format == "csv":
        rl_xg = _resolve_xg_helpers()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline().rstrip("\r\n").split(",")

        rows = pl.read_csv(
            path,
            null_values=getattr(rl_xg, "NULL_VALUES", []),
            schema_overrides=rl_xg.pbp_schema_overrides(
                set(header),
                rl_xg.model_pbp_columns(),
            ),
            infer_schema_length=10000,
        )
        scored = apply_xg_to_pbp(rows, xg_model_path)
        scored.write_csv(path)
        return

    raise ValueError("export_format must be one of: 'csv', 'parquet'")
