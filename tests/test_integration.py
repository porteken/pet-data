"""Integration test: full analytics pipeline with synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generate_analytics import (
    _load_pet_frame,
    generate_change_per_decade,
    generate_forecast,
    generate_percentiles,
)


def _write_shard_csv(
    root: Path, year: int, rows: list[dict[str, object]], city_shard: int = 0
) -> None:
    shard_dir = root / f"year={year}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(shard_dir / f"pet_batch_0000_{city_shard:02d}.parquet", index=False)


class TestEndToEndAnalyticsPipeline:
    """Simulate the CI generate-analytics job with synthetic PET shards."""

    @pytest.fixture()
    def pipeline_dirs(self, tmp_path: Path) -> dict[str, Path]:
        pet_root = tmp_path / "pet_data_csv"
        out_dir = tmp_path / "analytics_data_csv"
        rng = np.random.default_rng(0)

        # Create multi-year PET shards (2 city groups per year)
        for year in [2000, 2001, 2010, 2011]:
            for city_shard, loc_ids in enumerate([[10, 11], [20, 21]]):
                dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
                rows = [
                    {
                        "location_id": loc_id,
                        "date": d.strftime("%Y-%m-%d"),
                        "pet": round(rng.uniform(5, 35), 1),
                    }
                    for loc_id in loc_ids
                    for d in dates
                ]
                _write_shard_csv(pet_root, year, rows, city_shard)

        return {"pet_root": pet_root, "out_dir": out_dir}

    def test_load_pet_frame_from_shards(self, pipeline_dirs: dict[str, Path]) -> None:
        args = argparse.Namespace(
            pet_root=str(pipeline_dirs["pet_root"]),
            pet_csv="nonexistent.csv",
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert not df.empty
        assert "year" in df.columns
        assert set(df["year"].unique()) == {2000, 2001, 2010, 2011}

    def test_sharded_load_splits_files(self, pipeline_dirs: dict[str, Path]) -> None:
        args_0 = argparse.Namespace(
            pet_root=str(pipeline_dirs["pet_root"]),
            pet_csv="nonexistent.csv",
            shard_index=0,
            shard_count=2,
        )
        args_1 = argparse.Namespace(
            pet_root=str(pipeline_dirs["pet_root"]),
            pet_csv="nonexistent.csv",
            shard_index=1,
            shard_count=2,
        )
        df0 = _load_pet_frame(args_0)
        df1 = _load_pet_frame(args_1)
        # Together they should cover all 4 location IDs
        combined_ids = set(df0["location_id"]).union(set(df1["location_id"]))
        assert len(combined_ids) == 4

    def test_full_analytics_generation(self, pipeline_dirs: dict[str, Path]) -> None:
        args = argparse.Namespace(
            pet_root=str(pipeline_dirs["pet_root"]),
            pet_csv="nonexistent.csv",
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        out_dir = pipeline_dirs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Percentiles
        pct_path = generate_percentiles(df, out_dir)
        pct_df = pd.read_parquet(pct_path)
        assert len(pct_df) > 0
        assert (pct_df["p10"] <= pct_df["p90"]).all()

        # Forecast
        fc_path = generate_forecast(df, out_dir)
        fc_df = pd.read_parquet(fc_path)
        assert int(fc_df["year"].min()) == int(df["year"].max()) + 1  # pyright: ignore[reportArgumentType]
        assert int(fc_df["year"].max()) == 2100  # pyright: ignore[reportArgumentType]
        assert bool(fc_df["pet"].notna().all())

        # Change per decade
        chg_path = generate_change_per_decade(df, out_dir)
        chg_df = pd.read_parquet(chg_path)
        assert "year" in chg_df.columns
        assert "change" in chg_df.columns

    def test_empty_shard_beyond_file_count_returns_empty(
        self, pipeline_dirs: dict[str, Path]
    ) -> None:
        """Requesting a shard index beyond available files returns an empty frame."""
        # With 8 files total and shard_count=100, shard_index=99 gets 0 files
        args = argparse.Namespace(
            pet_root=str(pipeline_dirs["pet_root"]),
            pet_csv="nonexistent.csv",
            shard_index=99,
            shard_count=100,
        )
        df = _load_pet_frame(args)
        assert df.empty
