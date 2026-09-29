"""Final-model plumbing and the deliverable files (no CatBoost training)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from freight import config
from freight.final import (
    DECEMBER_OUT,
    FINAL_EXPERIMENT,
    MODEL_CARD_PATH,
    VALIDATION_OUT,
    ensemble_log_mean,
    format_december,
    format_validation,
    neutral_market_table,
    raw_december,
)

sys.path.insert(0, str(config.ROOT))
import score  # noqa: E402  (the organisers' scorer at the repo root)


# --------------------------------------------------------------------------- helpers
def test_ensemble_is_geometric_mean_of_rates() -> None:
    rates = np.array([[100.0, 400.0], [400.0, 100.0], [250.0, 250.0]])
    out = np.exp(ensemble_log_mean(np.log(rates)))
    np.testing.assert_allclose(out, np.cbrt(rates.prod(axis=0)))
    np.testing.assert_allclose(ensemble_log_mean(np.log(rates[:1])), np.log(rates[0]))


def test_neutral_market_only_flattens_december_index() -> None:
    idx = pd.date_range("2025-11-25", "2026-01-03", freq="D", name="date")
    market = pd.DataFrame({"mi_day_mean": np.linspace(0.8, 1.2, len(idx)), "qs_day_mean": 2.0}, index=idx)
    out = neutral_market_table(market)
    dec = (idx >= "2025-12-01") & (idx <= "2025-12-31")
    assert np.allclose(out.loc[dec, "mi_day_mean"], market.loc[dec, "mi_day_mean"].mean())
    pd.testing.assert_frame_equal(out.loc[~dec], market.loc[~dec])
    pd.testing.assert_series_equal(out["qs_day_mean"], market["qs_day_mean"])
    assert not np.allclose(market.loc[dec, "mi_day_mean"], out.loc[dec, "mi_day_mean"])  # input untouched


def test_format_validation_follows_template_order() -> None:
    template = pd.DataFrame({"load_id": ["TE-3", "TE-1", "TE-2"], "predicted_rate": np.nan})
    preds = pd.DataFrame({"load_id": ["TE-1", "TE-2", "TE-3"], "predicted_rate": [1.004, 2.555, 3.0]})
    out = format_validation(template, preds, decimals=2)
    assert list(out.columns) == ["load_id", "predicted_rate"]
    assert out["load_id"].tolist() == ["TE-3", "TE-1", "TE-2"]
    np.testing.assert_allclose(out["predicted_rate"], [3.0, 1.0, 2.56], atol=0.006)
    with pytest.raises(ValueError):
        format_validation(template, preds.iloc[:2])


def test_format_december_keeps_input_cells() -> None:
    raw = raw_december()
    out = format_december(raw, np.linspace(800, 900, len(raw)), decimals=2)
    assert list(out.columns) == list(raw.columns)
    pd.testing.assert_frame_equal(out.drop(columns="predicted_rate"), raw.drop(columns="predicted_rate"))
    score.validate_december(out)
    with pytest.raises(ValueError):
        format_december(raw, np.ones(3))


# --------------------------------------------------------------------------- deliverables
@pytest.mark.skipif(not VALIDATION_OUT.exists(), reason="run `make predict` first")
def test_validation_predictions_file() -> None:
    preds = pd.read_csv(VALIDATION_OUT)
    score.validate_predictions(preds)  # the organisers' own checks (SystemExit on failure)
    template = pd.read_csv(config.TEMPLATE_PATH)
    assert list(preds.columns) == list(template.columns)
    assert preds["load_id"].tolist() == template["load_id"].tolist()
    assert np.allclose(preds["predicted_rate"], preds["predicted_rate"].round(2))
    # header identical to the template, one line per load
    lines = VALIDATION_OUT.read_text().splitlines()
    assert lines[0] == "load_id,predicted_rate" and len(lines) == 12_001


@pytest.mark.skipif(not DECEMBER_OUT.exists(), reason="run `make predict` first")
def test_december_predictions_file() -> None:
    out = pd.read_csv(DECEMBER_OUT)
    score.validate_december(out)
    raw = pd.read_csv(config.DECEMBER_PATH)
    assert list(out.columns) == list(raw.columns)
    pd.testing.assert_frame_equal(out.drop(columns="predicted_rate"), raw.drop(columns="predicted_rate"))
    # every input cell is written back verbatim (dates in the input's format)
    for got, want in zip(DECEMBER_OUT.read_text().splitlines(), config.DECEMBER_PATH.read_text().splitlines()):
        assert got.rsplit(",", 1)[0] == want.rsplit(",", 1)[0]
    assert np.allclose(out["predicted_rate"], out["predicted_rate"].round(2))
    assert out["predicted_rate"].between(500, 1500).all()  # $1.4-4.2 per mile on a 360-mile lane


@pytest.mark.skipif(not MODEL_CARD_PATH.exists(), reason="run `make train` first")
def test_model_card_contract() -> None:
    card = json.loads(MODEL_CARD_PATH.read_text())
    best = json.loads(config.BEST_PARAMS_PATH.read_text())
    assert card["experiment_config"] == FINAL_EXPERIMENT
    assert card["catboost"]["tree_features"] == best["tree_features"]
    assert not set(card["catboost"]["tree_features"]) & (set(config.LEVEL_CB_EXCLUDED) | {"qs_dev"})
    for key, value in best["params"].items():
        assert card["catboost"]["params"][key] == value, key
    assert card["catboost"]["seeds"] == [0, 1, 2, 3, 4]
    assert list(card["level"]["coefficients"]) == ["const", "mi_day_mean", "t", "qe_ramp"]
    tr = card["training"]
    assert tr["n_rows_used"] + tr["n_excluded_corrupted"] == tr["n_rows_window"]
    assert 500 < tr["n_excluded_corrupted"] < 900  # ~1.4% corrupted labels
    assert tr["date_range"] == ["2025-01-01", "2025-10-31"]
    assert tr["outlier_threshold_log_ratio"] == pytest.approx(np.log(1.6))
