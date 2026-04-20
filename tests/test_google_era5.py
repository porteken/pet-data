"""Tests for google_era5.py — ERA5 batch/shard helpers and constants."""

from __future__ import annotations

import pathlib
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from typing_extensions import Self

import google_era5
from google_era5 import (
    DEFAULT_BATCH_HOURS,
    ERA5_TIME_ORIGIN,
    _approximate_dsrp,
    _compute_location_frame,
    _compute_pet_chunk,
    _hourly_flux_from_accumulation,
    _iter_time_batches,
    _normalize_longitudes_for_solar_geometry,
    _resolve_era5_max_workers,
    _run_era5_batch_jobs,
    _select_time_shard_batches,
    _wrap_longitudes_for_arco_selection,
    _year_time_slice,
)


class TestDefaultBatchHours:
    def test_is_monthly(self) -> None:
        assert DEFAULT_BATCH_HOURS == 24 * 30


class TestYearTimeSlice:
    def test_returns_tuple(self) -> None:
        start_h, end_h = _year_time_slice(2024)
        assert isinstance(start_h, int)
        assert isinstance(end_h, int)
        assert end_h > start_h

    def test_year_span_is_correct(self) -> None:
        start_h, end_h = _year_time_slice(2024)
        # 2024 is a leap year: 8784 hours
        assert end_h - start_h + 1 == 8784

    def test_starts_at_beginning_of_year(self) -> None:
        start_h, _ = _year_time_slice(2024)
        epoch = pd.Timestamp(google_era5.ERA5_TIME_ORIGIN)
        assert epoch + pd.Timedelta(hours=start_h) == pd.Timestamp(
            "2024-01-01 00:00:00"
        )

    def test_non_leap_year(self) -> None:
        start_h, end_h = _year_time_slice(2023)
        assert end_h - start_h + 1 == 8760


class TestIterTimeBatches:
    def test_covers_full_year(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        start_h, end_h = _year_time_slice(2024)
        assert batches[0][1] == start_h
        assert batches[-1][2] == end_h

    def test_batch_count_monthly(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=720)
        # 8784 / 720 = 12.2 → 13 batches
        assert len(batches) == 13

    def test_no_gaps_between_batches(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        for i in range(1, len(batches)):
            prev_end = batches[i - 1][2]
            curr_start = batches[i][1]
            assert curr_start == prev_end + 1

    def test_batch_indices_sequential(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        assert [b[0] for b in batches] == list(range(len(batches)))

    def test_raises_on_zero_batch_hours(self) -> None:
        with pytest.raises(ValueError, match="batch_hours"):
            _iter_time_batches(2024, batch_hours=0)


class TestSelectTimeShardBatches:
    def test_all_batches_assigned(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        all_assigned = []
        for shard_index in range(4):
            selected = _select_time_shard_batches(
                batches, time_shard_index=shard_index, time_shard_count=4
            )
            all_assigned.extend(selected)
        # Every batch should be assigned to exactly one shard
        assert sorted(b[0] for b in all_assigned) == sorted(b[0] for b in batches)

    def test_single_shard_returns_all(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        selected = _select_time_shard_batches(
            batches, time_shard_index=0, time_shard_count=1
        )
        assert len(selected) == len(batches)

    def test_four_shards_balanced(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        shard_sizes = []
        for shard_index in range(4):
            selected = _select_time_shard_batches(
                batches, time_shard_index=shard_index, time_shard_count=4
            )
            shard_sizes.append(len(selected))
        # No shard should differ by more than 1 from another
        assert max(shard_sizes) - min(shard_sizes) <= 1

    def test_raises_on_invalid_index(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        with pytest.raises(ValueError, match="time_shard_index"):
            _select_time_shard_batches(batches, time_shard_index=4, time_shard_count=4)


class TestResolveEra5MaxWorkers:
    def test_minus_one_returns_cpu_count(self) -> None:
        result = _resolve_era5_max_workers(-1)
        assert result >= 1

    def test_explicit_value_passes_through(self) -> None:
        assert _resolve_era5_max_workers(8) == 8

    def test_raises_on_zero(self) -> None:
        with pytest.raises(ValueError, match="max_workers"):
            _resolve_era5_max_workers(0)


class TestHourlyFluxFromAccumulation:
    def test_converts_joules_per_square_meter_to_watts_per_square_meter(self) -> None:
        converted = _hourly_flux_from_accumulation(np.array([3600.0, 7200.0, 1800.0]))

        np.testing.assert_allclose(converted, np.array([1.0, 2.0, 0.5]))


class TestApproximateDsrp:
    def test_does_not_cap_direct_normal_radiation_to_ssrd(self) -> None:
        thermofeel = MagicMock()
        thermofeel.approximate_dsrp.return_value = np.array([500.0, 300.0, 100.0])

        result = _approximate_dsrp(
            thermofeel,
            fdir_flux=np.array([50.0, 60.0, 70.0]),
            cossza=np.array([0.5, 0.2, 0.05]),
            _ssrd_flux=np.array([100.0, 80.0, 90.0]),
        )

        np.testing.assert_allclose(result, np.array([500.0, 300.0, 0.0]))


class TestLongitudeNormalization:
    def test_wrap_longitudes_for_arco_selection_maps_negative_us_longitudes(
        self,
    ) -> None:
        wrapped = _wrap_longitudes_for_arco_selection(
            np.array([-104.75, -80.5, -74.0, 0.0, 179.75, 360.0])
        )

        np.testing.assert_allclose(
            wrapped,
            np.array([255.25, 279.5, 286.0, 0.0, 179.75, 0.0]),
        )

    def test_normalize_longitudes_for_solar_geometry_restores_western_hemisphere(
        self,
    ) -> None:
        normalized = _normalize_longitudes_for_solar_geometry(
            np.array([255.25, 279.5, 286.0, 0.0, 179.75, 181.0])
        )

        np.testing.assert_allclose(
            normalized,
            np.array([-104.75, -80.5, -74.0, 0.0, 179.75, -179.0]),
        )


class TestComputePetChunk:
    def test_discards_pet_values_above_sixty_c(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df = pd.DataFrame(
            {
                "t": [25.0, 26.0, 27.0],
                "mrt": [30.0, 31.0, 32.0],
                "v": [1.0, 1.0, 1.0],
                "rh": [50.0, 50.0, 50.0],
            }
        )

        monkeypatch.setattr(
            google_era5,
            "pet_corrected",
            lambda *_args, **_kwargs: np.array([22.0, 61.0, -55.0]),
        )

        result = _compute_pet_chunk(df)

        assert result["pet"].tolist() == [22.0]


class TestRunEra5BatchJobs:
    def test_parallel_branch_uses_threads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        shard_df = pd.DataFrame(
            {
                "location_id": [1],
                "lat": [35.0],
                "lng": [-80.0],
            }
        )
        completed_batches: list[int] = []

        def fake_pet_batch_exists(
            _pet_root: str,
            _year: int,
            _city_shard_index: int,
            _batch_index: int,
        ) -> bool:
            return False

        def fake_process_batch(**kwargs: object) -> int:
            batch_index = cast("int", kwargs["batch_index"])
            completed_batches.append(batch_index)
            return batch_index

        monkeypatch.setattr(
            google_era5,
            "_pet_batch_exists",
            fake_pet_batch_exists,
        )
        monkeypatch.setattr(
            google_era5,
            "_process_era5_batch_with_thread_dataset",
            fake_process_batch,
        )
        monkeypatch.setenv("ERA5_CONCURRENCY_PROFILE", "aggressive")

        wrote_batches = _run_era5_batch_jobs(
            selected_batches=[(0, 0, 23), (1, 24, 47)],
            shard_df=shard_df,
            era5_root=str(tmp_path / "era5"),
            year=2001,
            city_shard_index=0,
            city_shard_count=1,
            time_shard_index=0,
            time_shard_count=1,
            max_workers=2,
            force=False,
        )

        assert wrote_batches is True
        assert sorted(completed_batches) == [0, 1]


class TestComputeLocationFrameAlignment:
    """Verify that weather values are correctly aligned to (location, time) pairs.

    ERA5 xarray selection produces (n_times, n_locs) arrays.  Without a .T
    transpose those arrays are raveled in time-first order, which mismatches the
    DataFrame's loc-first row ordering (np.repeat / np.tile).  The test below
    constructs two synthetic cities whose daily-max PET ranks should be strictly
    ordered (hot city always > cold city) - a property that breaks when loc/time
    data is scrambled.
    """

    def _make_fake_dataset(
        self,
        n_times: int,
        n_locs: int,
        hot_temp_k: float,
        cold_temp_k: float,
    ) -> MagicMock:
        """Return a minimal xarray-like Dataset mock with two distinct temperature profiles.

        City 0 (hot): constant ``hot_temp_k``, dew-point 10 K below.
        City 1 (cold): constant ``cold_temp_k``, dew-point 10 K below.
        All radiation variables are zero (night-time → no solar MRT boost).
        Wind speed is fixed at 2 m/s (> 0 so the filter keeps all rows).
        """
        # Temperatures: shape (n_times, n_locs) — xarray returns time-first
        temps = np.empty((n_times, n_locs), dtype=float)
        temps[:, 0] = hot_temp_k
        temps[:, 1] = cold_temp_k
        dew = temps - 10.0  # dew-point 10 K below air temp

        # Wind components: shape (n_times, n_locs)
        np.full((n_times, n_locs), np.sqrt(2.0))  # u=v=√2 → speed=2

        # Use hourly ERA5-style accumulations (J/m²) that convert to moderate fluxes.
        rad_down = np.full((n_times, n_locs), 300.0 * 3600.0)
        rad_net = np.zeros((n_times, n_locs))

        # Build xarray-style variables accessible by key
        raw = {
            "10u": temps * 0 + np.sqrt(2.0),
            "10v": temps * 0 + np.sqrt(2.0),
            "2t": temps,
            "2d": dew,
            "ssrd": rad_net.copy(),
            "strd": rad_down.copy(),
            "ssr": rad_net.copy(),
            "str": rad_net.copy(),
            "fdir": rad_net.copy(),
            "msdrswrf": rad_net.copy(),
        }

        # time coordinate: integer hours since ERA5_TIME_ORIGIN for a Jan week
        epoch = pd.Timestamp(ERA5_TIME_ORIGIN)
        start_h = int((pd.Timestamp("2010-01-01") - epoch).total_seconds() // 3600)
        time_vals = np.arange(start_h, start_h + n_times, dtype=int)

        class _FakeVar:
            def __init__(self, arr: np.ndarray) -> None:
                self.values = arr

        class _FakeDataset:
            def __init__(self) -> None:
                self._raw = raw
                self.time = _FakeVar(time_vals)

            def __getitem__(self, key: str) -> _FakeVar:
                return _FakeVar(self._raw[key])

            def assign_coords(self, **kwargs: object) -> _FakeDataset:
                copy = _FakeDataset()
                if "time" in kwargs:
                    copy.time = _FakeVar(np.asarray(kwargs["time"]))
                return copy

        return _FakeDataset()  # type: ignore[return-value]

    def test_hot_city_has_higher_pet_than_cold_city(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """City 0 (hot) should consistently produce higher daily max PET than city 1 (cold)."""
        n_times = 24  # one day of hourly data
        n_locs = 2
        hot_k = 308.15  # 35 °C
        cold_k = 278.15  # 5 °C

        fake_ds = self._make_fake_dataset(n_times, n_locs, hot_k, cold_k)

        # Patch dask so compute() returns the mock object unchanged
        import dask as _dask

        class _FakeDaskCtx:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_: object) -> None:
                # The fake context has no resources to release.
                pass

        monkeypatch.setattr(
            _dask, "config", MagicMock(set=lambda **_kw: _FakeDaskCtx())
        )

        # Patch xr.DataArray so sel() works — we bypass xarray's sel entirely
        # by replacing city_selection.compute() return value via monkeypatching
        # _compute_location_frame's internal calls.

        # We intercept at the compute() call inside _compute_location_frame.
        # The easiest way: patch the Dataset's sel to return an object whose
        # .compute() returns our fake dataset.
        class _FakeSel:
            def sel(self, *_args: object, **_kw: object) -> _FakeSel:
                return self

            def compute(self) -> object:
                return fake_ds

        class _FakeSliceable:
            def __getitem__(self, _keys: object) -> _FakeSliceable:
                return self

            def sel(self, *_args: object, **_kw: object) -> _FakeSel:
                return _FakeSel()

        fake_ds_with_slice = _FakeSliceable()

        cities = pd.DataFrame(
            {
                "location_id": [0, 1],
                "lat": [40.0, 40.0],
                "lng": [-75.0, -75.0],
            }
        )

        result = _compute_location_frame(
            fake_ds_with_slice,  # type: ignore[arg-type]
            cities,
            start_h=0,
            end_h=n_times - 1,
            compute_workers=1,
        )

        assert not result.empty, "Expected non-empty PET result"
        # After the transposition fix each city should receive its own temperature
        pet_hot = result.loc[result["location_id"] == 0, "pet"].max()
        pet_cold = result.loc[result["location_id"] == 1, "pet"].max()
        assert pet_hot > pet_cold, (
            f"Hot city PET ({pet_hot}) should exceed cold city PET ({pet_cold}). "
            "This failure typically means the (n_times, n_locs) arrays are being "
            "raveled without transposition, scrambling loc/time alignment."
        )
