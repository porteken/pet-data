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
    root: Path, year: int, tile_id: int, rows: list[dict[str, object]]
) -> None:
    shard_dir = root / f"year={year}" / f"tile_id={tile_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(shard_dir / "pet.csv.gz", index=False)


class TestEndToEndAnalyticsPipeline:
    """Simulate the CI generate-analytics job with synthetic PET shards."""

    @pytest.fixture()
    def pipeline_dirs(self, tmp_path: Path) -> dict[str, Path]:
        pet_root = tmp_path / "pet_data_csv"
        out_dir = tmp_path / "analytics_data_csv"
        rng = np.random.default_rng(0)

        # Create multi-year, multi-tile PET shards
        for year in [2000, 2001, 2010, 2011]:
            for tile_id in [1, 2]:
                dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
                rows = [
                    {
                        "location_id": loc_id,
                        "date": d.strftime("%Y-%m-%d"),
                        "pet": round(rng.uniform(5, 35), 1),
                    }
                    for loc_id in [tile_id * 10, tile_id * 10 + 1]
                    for d in dates
                ]
                _write_shard_csv(pet_root, year, tile_id, rows)

        return {"pet_root": pet_root, "out_dir": out_dir}

    def test_load_pet_frame_from_shards(self, pipeline_dirs: dict[str, Path]) -> None:
        args = argparse.Namespace(
            pet_root=str(pipeline_dirs["pet_root"]),
            pet_csv="nonexistent.csv.gz",
            tile_ids=None,
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert not df.empty
        assert "year" in df.columns
        assert set(df["year"].unique()) == {2000, 2001, 2010, 2011}

    def test_sharded_load_splits_tiles(self, pipeline_dirs: dict[str, Path]) -> None:
        args_0 = argparse.Namespace(
            pet_root=str(pipeline_dirs["pet_root"]),
            pet_csv="nonexistent.csv.gz",
            tile_ids=None,
            shard_index=0,
            shard_count=2,
        )
        args_1 = argparse.Namespace(
            pet_root=str(pipeline_dirs["pet_root"]),
            pet_csv="nonexistent.csv.gz",
            tile_ids=None,
            shard_index=1,
            shard_count=2,
        )
        df0 = _load_pet_frame(args_0)
        df1 = _load_pet_frame(args_1)
        combined_ids = set(df0["location_id"]).union(set(df1["location_id"]))
        assert len(combined_ids) == 4  # 2 tiles × 2 locations each
        assert set(df0["location_id"]).isdisjoint(set(df1["location_id"]))

    def test_full_analytics_generation(self, pipeline_dirs: dict[str, Path]) -> None:
        args = argparse.Namespace(
            pet_root=str(pipeline_dirs["pet_root"]),
            pet_csv="nonexistent.csv.gz",
            tile_ids=None,
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        out_dir = pipeline_dirs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Percentiles
        pct_path = generate_percentiles(df, out_dir)
        pct_df = pd.read_csv(pct_path)
        assert len(pct_df) > 0
        assert (pct_df["p10"] <= pct_df["p90"]).all()

        # Forecast
        fc_path = generate_forecast(df, out_dir)
        fc_df = pd.read_csv(fc_path)
        assert set(fc_df["year"]) == {2030, 2040, 2050}
        assert bool(fc_df["pet"].notna().all())

        # Change per decade
        chg_path = generate_change_per_decade(df, out_dir)
        chg_df = pd.read_csv(chg_path)
        assert "decade" in chg_df.columns
        assert "change" in chg_df.columns

    def test_empty_shard_match_returns_empty(
        self, pipeline_dirs: dict[str, Path]
    ) -> None:
        """Requesting a tile that doesn't exist returns an empty frame."""
        args = argparse.Namespace(
            pet_root=str(pipeline_dirs["pet_root"]),
            pet_csv="nonexistent.csv.gz",
            tile_ids=[999],
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert df.empty
