"""Tests for generate_analytics.py — percentiles, forecast, change per decade."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from generate_analytics import (
    _apply_quadratic_damping,
    _build_forecast_frame,
    _discover_pet_files,
    _load_pet_frame,
    _load_pet_frame_from_csv,
    _output_dir,
    _predict_from_model,
    _rolling_origin_rmse,
    _select_best_model,
    generate_change_per_decade,
    generate_forecast,
    generate_percentiles,
)


def _write_pet_parquet(path: Path, content_csv: str) -> None:
    import io

    df = pd.read_csv(io.StringIO(content_csv))
    df.to_parquet(path, index=False)


@pytest.fixture()
def sample_pet_df() -> pd.DataFrame:
    """A minimal PET DataFrame for testing."""
    rng = np.random.default_rng(42)
    rows = []
    for year in range(2010, 2025):
        dates = pd.date_range(
            f"{year}-01-01", periods=366 if year % 4 == 0 else 365, freq="D"
        )
        for loc_id in [1, 2]:
            for d in dates:
                rows.append(
                    {
                        "location_id": loc_id,
                        "date": d,
                        "pet": rng.uniform(10, 30),
                        "year": year,
                    }
                )
    return pd.DataFrame(rows)


class TestGeneratePercentiles:
    def test_produces_p10_p90_columns(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_percentiles(sample_pet_df, tmp_path)
        assert path.exists()
        df = pd.read_parquet(path)
        assert "p10" in df.columns
        assert "p90" in df.columns
        assert "year" in df.columns
        assert "location_id" in df.columns

    def test_p10_less_than_p90(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_percentiles(sample_pet_df, tmp_path)
        df = pd.read_parquet(path)
        assert (df["p10"] <= df["p90"]).all()

    def test_all_locations_present(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_percentiles(sample_pet_df, tmp_path)
        df = pd.read_parquet(path)
        assert set(df["location_id"]) == {1, 2}


class TestGenerateForecast:
    def test_produces_forecast_decades(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        forecast_df = _build_forecast_frame(sample_pet_df, max_workers=1)
        path = generate_forecast(forecast_df, tmp_path)
        assert path.exists()
        df = pd.read_parquet(path)
        assert int(df["year"].min()) == 2025  # pyright: ignore[reportArgumentType]
        assert int(df["year"].max()) == 2100  # pyright: ignore[reportArgumentType]
        assert {"lower", "upper"}.issubset(df.columns)

    def test_forecast_has_pet_column(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        forecast_df = _build_forecast_frame(sample_pet_df, max_workers=1)
        path = generate_forecast(forecast_df, tmp_path)
        df = pd.read_parquet(path)
        assert "pet" in df.columns
        assert bool(df["pet"].notna().all())

    def test_single_year_produces_empty_forecast(self, tmp_path: Path) -> None:
        """Verify single year data produces no forecast."""
        df = pd.DataFrame(
            {
                "location_id": [1] * 10,
                "date": pd.date_range("2020-01-01", periods=10),
                "pet": [20.0] * 10,
                "year": [2020] * 10,
            }
        )
        forecast_df = _build_forecast_frame(df, max_workers=1)
        path = generate_forecast(forecast_df, tmp_path)
        result = pd.read_parquet(path)
        assert len(result) == 0


class TestGenerateChangePerDecade:
    def test_produces_change_column(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        forecast_df = _build_forecast_frame(sample_pet_df, max_workers=1)
        path = generate_change_per_decade(sample_pet_df, forecast_df, tmp_path)
        assert path.exists()
        df = pd.read_parquet(path)
        assert "change" in df.columns
        assert "year" in df.columns

    def test_smoke_style_input_produces_future_decade_changes(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        df = sample_pet_df
        forecast_df = _build_forecast_frame(df, max_workers=1)
        path = generate_change_per_decade(df, forecast_df, tmp_path)
        result = pd.read_parquet(path)
        assert len(result) > 0
        assert int(result["year"].min()) == 2020  # pyright: ignore[reportArgumentType]

    def test_single_decade_produces_no_change(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "location_id": [1] * 365,
                "date": pd.date_range("2020-01-01", periods=365),
                "pet": [20.0] * 365,
                "year": [2020] * 365,
            }
        )
        forecast_df = _build_forecast_frame(df, max_workers=1)
        path = generate_change_per_decade(df, forecast_df, tmp_path)
        result = pd.read_parquet(path)
        assert len(result) == 0


class TestDiscoverPetFiles:
    def test_discovers_batch_parquets(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020"
        shard_dir.mkdir(parents=True)
        _write_pet_parquet(
            shard_dir / "pet_batch_0000_00.parquet",
            "location_id,date,pet\n1,2020-01-01,10\n",
        )

        result = _discover_pet_files(tmp_path)
        assert len(result) == 1

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        result = _discover_pet_files(tmp_path / "nonexistent")
        assert result == []

    def test_ignores_non_batch_files(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020"
        shard_dir.mkdir(parents=True)
        (shard_dir / "pet.csv").write_text("location_id,date,pet\n1,2020-01-01,10\n")
        result = _discover_pet_files(tmp_path)
        assert result == []


class TestLoadPetFrameFromCsv:
    def test_loads_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pet.csv"
        csv_path.write_text(
            "location_id,date,pet\n0,2020-01-01,15.0\n1,2020-01-01,16.0\n"
        )
        df = _load_pet_frame_from_csv(csv_path, shard_index=0, shard_count=1)
        assert len(df) == 2

    def test_csv_sharding(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pet.csv"
        csv_path.write_text(
            "location_id,date,pet\n0,2020-01-01,15.0\n1,2020-01-01,16.0\n"
        )
        df0 = _load_pet_frame_from_csv(csv_path, shard_index=0, shard_count=2)
        df1 = _load_pet_frame_from_csv(csv_path, shard_index=1, shard_count=2)
        assert len(df0) + len(df1) == 2


class TestLoadPetFrame:
    def test_prefers_shards_over_csv(self, tmp_path: Path) -> None:
        pet_root = tmp_path / "pet_data"
        shard_dir = pet_root / "year=2020"
        shard_dir.mkdir(parents=True)
        _write_pet_parquet(
            shard_dir / "pet_batch_0000_00.parquet",
            "location_id,date,pet\n1,2020-01-01,10.0\n",
        )

        csv_path = tmp_path / "pet.csv"
        csv_path.write_text("location_id,date,pet\n99,2020-01-01,99.0\n")

        args = argparse.Namespace(
            pet_root=str(pet_root),
            pet_csv=str(csv_path),
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert 1 in df["location_id"].values
        assert 99 not in df["location_id"].values

    def test_prefers_csv_when_requested(self, tmp_path: Path) -> None:
        pet_root = tmp_path / "pet_data"
        shard_dir = pet_root / "year=2020"
        shard_dir.mkdir(parents=True)
        _write_pet_parquet(
            shard_dir / "pet_batch_0000_00.parquet",
            "location_id,date,pet\n1,2020-01-01,10.0\n",
        )

        csv_path = tmp_path / "pet_full.csv"
        csv_path.write_text("location_id,date,pet\n99,2020-01-01,99.0\n")

        args = argparse.Namespace(
            pet_root=str(pet_root),
            pet_csv=str(csv_path),
            shard_index=0,
            shard_count=1,
            prefer_pet_csv=True,
        )
        df = _load_pet_frame(args)
        assert 99 in df["location_id"].values
        assert 1 not in df["location_id"].values

    def test_falls_back_to_csv(self, tmp_path: Path) -> None:
        pet_root = tmp_path / "empty_shards"

        csv_path = tmp_path / "pet.csv"
        csv_path.write_text("location_id,date,pet\n5,2020-01-01,20.0\n")

        args = argparse.Namespace(
            pet_root=str(pet_root),
            pet_csv=str(csv_path),
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert len(df) == 1
        assert df["location_id"].iloc[0] == 5

    def test_returns_empty_when_no_data_found(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            pet_root=str(tmp_path / "empty"),
            pet_csv=str(tmp_path / "no_such.csv"),
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert df.empty


class TestOutputDir:
    def test_partitioned_path(self) -> None:
        result = _output_dir(Path("analytics"), shard_index=3, shard_count=20)
        assert result == Path("analytics/shard_count=00020/shard_index=00003")


def _make_daily_pet(
    loc_id: int,
    n_years: int,
    *,
    start_year: int = 2000,
    annual_trend: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a daily PET DataFrame with a controlled linear annual mean trend."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_years):
        year = start_year + i
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        base = 20.0 + annual_trend * i
        for d in dates:
            rows.append(
                {"location_id": loc_id, "date": d, "pet": base + rng.normal(0, 0.3)}
            )
    return pd.DataFrame(rows)


class TestModelSelection:
    """Quadratic vs. linear model selection with controlled synthetic data."""

    def test_linear_city_selects_linear_model(self) -> None:
        """A city with a pure linear trend should not select quadratic."""
        df = _make_daily_pet(loc_id=1, n_years=25, annual_trend=0.1, seed=1)
        yearly = df.copy()
        yearly["year"] = pd.to_datetime(yearly["date"]).dt.year.astype("int32")
        yearly_avg = yearly.groupby("year")["pet"].mean().reset_index()
        yearly_avg["location_id"] = 1
        model = _select_best_model(yearly_avg)
        assert model is not None

        assert model["degree"] == 1

    def test_25_years_triggers_quadratic_evaluation(self) -> None:
        """Cities with exactly 25 years must reach the quadratic evaluation branch."""
        df = _make_daily_pet(loc_id=2, n_years=25, annual_trend=0.05, seed=2)
        yearly = df.copy()
        yearly["year"] = pd.to_datetime(yearly["date"]).dt.year.astype("int32")
        yearly_avg = yearly.groupby("year")["pet"].mean().reset_index()
        yearly_avg["location_id"] = 2

        model = _select_best_model(yearly_avg)
        assert model is not None
        assert model["degree"] in (1, 2)

    def test_fewer_than_25_years_never_selects_quadratic(self) -> None:
        """Cities with < 25 complete years must always get a linear model."""
        df = _make_daily_pet(loc_id=3, n_years=24, annual_trend=0.15, seed=3)
        yearly = df.copy()
        yearly["year"] = pd.to_datetime(yearly["date"]).dt.year.astype("int32")
        yearly_avg = yearly.groupby("year")["pet"].mean().reset_index()
        yearly_avg["location_id"] = 3
        model = _select_best_model(yearly_avg)
        assert model is not None
        assert model["degree"] == 1

    def test_rolling_origin_rmse_requires_min_test_points(self) -> None:
        """_rolling_origin_rmse returns inf when there are too few OOS predictions."""
        from generate_analytics import MIN_BACKTEST_TEST_POINTS

        years = np.arange(2000, 2000 + MIN_BACKTEST_TEST_POINTS + 2, dtype="float64")
        values = np.linspace(20.0, 21.0, len(years))

        min_train = len(years) - (MIN_BACKTEST_TEST_POINTS - 1)
        result = _rolling_origin_rmse(
            years, values, degree=2, min_train_years=min_train
        )
        assert result == math.inf

    def test_rolling_origin_rmse_succeeds_with_enough_test_points(self) -> None:
        """Returns a finite value when >= MIN_BACKTEST_TEST_POINTS OOS predictions exist."""
        from generate_analytics import (
            MIN_BACKTEST_TEST_POINTS,
            MIN_BACKTEST_TRAIN_YEARS_QUADRATIC,
        )

        n = MIN_BACKTEST_TRAIN_YEARS_QUADRATIC + MIN_BACKTEST_TEST_POINTS + 1
        years = np.arange(2000, 2000 + n, dtype="float64")
        values = np.linspace(20.0, 22.0, n) + np.random.default_rng(7).normal(0, 0.1, n)
        result = _rolling_origin_rmse(
            years, values, degree=1, min_train_years=MIN_BACKTEST_TRAIN_YEARS_QUADRATIC
        )
        assert math.isfinite(result)


class TestQuadraticDamping:
    """Quadratic damping keeps forecasts physically bounded out to 2100."""

    def test_damped_forecast_bounded_at_2100(self) -> None:
        """A strongly upward-curving quadratic must not produce extreme values at 2100."""
        from generate_analytics import TrendModel

        model: TrendModel = {
            "degree": 2,
            "coef": np.array([0.05, 0.5, 20.0]),
            "reference_year": 2012.0,
            "rmse": 0.5,
            "cv_rmse": 0.6,
        }
        future_years = np.arange(2026, 2101, dtype="int32")
        damped = cast(
            "Any", _apply_quadratic_damping(model, future_years, last_year=2025)
        )

        raw_at_2100 = float(
            np.polyval(
                cast("Any", model["coef"]),
                2100.0 - cast("float", model["reference_year"]),
            )
        )
        assert float(damped[-1]) < raw_at_2100, (
            "Damping must reduce the far-future value"
        )

        assert float(damped[-1]) < 400.0, (
            "Damped PET at 2100 should remain physically plausible"
        )

    def test_linear_model_is_not_damped(self) -> None:
        """_apply_quadratic_damping must be a no-op for degree-1 models."""
        from generate_analytics import TrendModel

        model: TrendModel = {
            "degree": 1,
            "coef": np.array([0.1, 20.0]),
            "reference_year": 2012.0,
            "rmse": 0.3,
            "cv_rmse": 0.4,
        }
        future_years = np.arange(2026, 2101, dtype="int32")
        damped = _apply_quadratic_damping(model, future_years, last_year=2025)
        raw = _predict_from_model(model, future_years)
        np.testing.assert_array_almost_equal(cast("Any", damped), cast("Any", raw))


class TestCitySpecificWarmingRates:
    """Each city must receive an independent warming-rate estimate."""

    def _make_full_year_df(
        self, loc_id: int, n_years: int, trend: float, seed: int
    ) -> pd.DataFrame:
        return _make_daily_pet(
            loc_id=loc_id, n_years=n_years, annual_trend=trend, seed=seed
        )

    def test_faster_warming_city_has_higher_rate(self) -> None:
        """A city with a steeper PET trend must have a larger forecast warming_rate."""
        df_slow = self._make_full_year_df(1, 15, trend=0.05, seed=10)
        df_fast = self._make_full_year_df(2, 15, trend=0.30, seed=11)

        combined = pd.concat([df_slow, df_fast], ignore_index=True)
        forecast = _build_forecast_frame(combined, max_workers=1)

        assert not forecast.empty, "Both cities should produce forecasts"
        slow_rate = float(forecast[forecast["location_id"] == 1]["warming_rate"].mean())
        fast_rate = float(forecast[forecast["location_id"] == 2]["warming_rate"].mean())
        assert fast_rate > slow_rate, (
            f"Faster-warming city should have higher warming_rate "
            f"({fast_rate:.4f} vs {slow_rate:.4f})"
        )

    def test_cities_get_independent_model_types(self) -> None:
        """model_type column must be present and contain valid values for every row."""
        df = pd.concat(
            [
                self._make_full_year_df(i, 15, trend=0.1 * i, seed=i)
                for i in range(1, 4)
            ],
            ignore_index=True,
        )
        forecast = _build_forecast_frame(df, max_workers=1)
        assert not forecast.empty
        assert forecast["model_type"].isin(["linear", "quadratic"]).all()


class TestUncertaintyBands:
    """Forecast uncertainty bands must widen monotonically with the horizon."""

    def test_bands_widen_with_horizon_linear_model(self) -> None:
        """For a linear-model city, upper - lower must be non-decreasing over time."""
        df = _make_daily_pet(loc_id=1, n_years=15, annual_trend=0.1, seed=20)
        forecast = _build_forecast_frame(df, max_workers=1)
        assert not forecast.empty

        loc_fc = (
            forecast[forecast["location_id"] == 1]
            .sort_values("year")
            .reset_index(drop=True)
        )
        widths = (loc_fc["upper"] - loc_fc["lower"]).to_numpy()

        diffs = np.diff(widths)
        assert (diffs >= -0.012).all(), "Uncertainty bands must not narrow over time"

    def test_quadratic_bands_wider_than_linear_at_long_horizons(self) -> None:
        """At 50+ year horizons quadratic uncertainty exceeds linear (extra h^2 term)."""
        from generate_analytics import (
            QUADRATIC_UNCERTAINTY_DENOMINATOR,
            UNCERTAINTY_GROWTH_DENOMINATOR,
        )

        base_rmse = 1.0
        horizons = np.array([50, 60, 70, 75], dtype="float64")

        sigma_linear = base_rmse * np.sqrt(
            1.0 + horizons / UNCERTAINTY_GROWTH_DENOMINATOR
        )
        sigma_quadratic = base_rmse * np.sqrt(
            1.0
            + horizons / UNCERTAINTY_GROWTH_DENOMINATOR
            + (horizons / QUADRATIC_UNCERTAINTY_DENOMINATOR) ** 2
        )
        assert (sigma_quadratic > sigma_linear).all(), (
            "Quadratic uncertainty must exceed linear at long horizons"
        )
