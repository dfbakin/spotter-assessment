"""Phase-2b tuning plumbing: tuned-params file, search space, ablation / selection rules (tiny fits only)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import optuna
import pandas as pd
import pytest
from catboost import CatBoostRegressor, Pool

from freight import config
from freight.backtest import EXPERIMENTS, BacktestData, _feature_view, catboost_params
from freight.features import build_features
from freight.models import DEFAULT_CATBOOST_PARAMS, load_tuned_params
from freight.tuning import DEFAULT_TRIAL_PARAMS, choose, suggest_params, summarise_ablation


def _tiny_fit(params: dict) -> CatBoostRegressor:
    """Fit CatBoost with ``params`` merged over the project defaults on a toy frame (3 trees)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=200), "equipment": rng.choice(["Dry Van", "Reefer", "Flatbed"], 200)})
    y = X["x"] + rng.normal(0, 0.1, 200)
    model = CatBoostRegressor(**{**DEFAULT_CATBOOST_PARAMS, **params, "iterations": 3})
    return model.fit(Pool(X, y, cat_features=["equipment"]))


def test_load_tuned_params_roundtrip(tmp_path) -> None:
    params = {"depth": 5, "learning_rate": 0.1, "iterations": 700, "bootstrap_type": "Bernoulli", "subsample": 0.7}
    path = tmp_path / "best_params.json"
    path.write_text(json.dumps({"params": params, "scores": {}}))
    assert load_tuned_params(path) == params
    _tiny_fit(load_tuned_params(path))

    cfg = replace(EXPERIMENTS["cb_level_tuned"], params_file=str(path))  # absolute path wins over ROOT
    assert catboost_params(cfg, 1) == {**params, "random_seed": 1}

    path.write_text(json.dumps({"params": {**params, "random_seed": 3}}))
    with pytest.raises(ValueError):
        load_tuned_params(path)
    with pytest.raises(FileNotFoundError):
        load_tuned_params(tmp_path / "missing.json")


@pytest.mark.skipif(not config.BEST_PARAMS_PATH.exists(), reason="run `make tune` first")
def test_best_params_file_loads_into_model_config(real_train: pd.DataFrame, real_market: pd.DataFrame) -> None:
    blob = json.loads(config.BEST_PARAMS_PATH.read_text())
    cfg = EXPERIMENTS["cb_level_tuned"]
    params = catboost_params(cfg, 0)
    assert params == {**blob["params"], "random_seed": 0}
    assert {"depth", "learning_rate", "iterations"} <= set(params)
    _tiny_fit(params)  # every key is a valid CatBoost parameter and the combination is legal
    # the features the tuned params were selected with are the ones cb_level_tuned trains on
    X, cats = build_features(real_train.head(300), real_market, use_time=True)
    data = BacktestData(frame=real_train.head(300), X=X, cat_features=cats, market_table=real_market)
    assert list(_feature_view(cfg, data)[0].columns) == blob["tree_features"]
    assert not set(blob["tree_features"]) & set(config.LEVEL_CB_EXCLUDED)


def test_defaults_are_a_point_of_the_search_space() -> None:
    params = suggest_params(optuna.trial.FixedTrial(DEFAULT_TRIAL_PARAMS))
    assert params == DEFAULT_TRIAL_PARAMS
    for key in ("depth", "learning_rate", "iterations", "l2_leaf_reg"):
        assert params[key] == DEFAULT_CATBOOST_PARAMS[key]
    # CatBoost's own CPU defaults for the remaining keys
    implicit = _tiny_fit({}).get_all_params()
    assert implicit["bootstrap_type"] == params["bootstrap_type"]
    assert implicit["subsample"] == pytest.approx(params["subsample"])
    for key in ("border_count", "one_hot_max_size", "grow_policy", "random_strength"):
        assert implicit[key] == params[key], key


def test_search_space_conditionals() -> None:
    fixed = {**DEFAULT_TRIAL_PARAMS, "bootstrap_type": "Bayesian", "grow_policy": "Depthwise"}
    fixed.pop("subsample")
    fixed.update(bagging_temperature=1.5, min_data_in_leaf=20)
    params = suggest_params(optuna.trial.FixedTrial(fixed))
    assert "subsample" not in params and params["bagging_temperature"] == 1.5 and params["min_data_in_leaf"] == 20
    _tiny_fit(params)


def test_ablation_decision_rule() -> None:
    rows = []
    for variant, f1, f2 in [("full", 30.0, 32.0), ("lane", 28.5, 31.0), ("city", 29.5, 31.5), ("dow_dom", 33.0, 34.0)]:
        for fold, v in (("F1", f1), ("F2", f2), ("F3", 20.0)):
            rows.append({"variant": variant, "dropped": variant, "n_features": 10, "fold": fold, "mae_debiased": v, "mae_clean": v + 10, "mae_all": v + 60})
    out = summarise_ablation(pd.DataFrame(rows)).set_index("variant")
    assert out.loc["lane", "delta_mae_debiased_f12"] == pytest.approx(-1.25)
    assert out.loc["lane", "decision"] == "drop"  # improves by more than $1
    assert out.loc["city", "decision"] == "keep"  # improves by only $0.5
    assert out.loc["dow_dom", "decision"] == "keep"


def test_choose_prefers_fastest_within_tolerance() -> None:
    summary = pd.DataFrame(
        {
            "candidate": ["a", "b", "c"],
            "mae_debiased_f12_mean": [30.0, 30.4, 31.0],
            "fit_seconds_f12": [90.0, 20.0, 5.0],
        }
    )
    assert choose(summary, tol=0.5)["candidate"] == "b"
    assert choose(summary, tol=0.1)["candidate"] == "a"
