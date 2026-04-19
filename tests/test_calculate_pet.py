"""Tests for calculate_pet.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

import calculate_pet

if TYPE_CHECKING:
    import pytest


class TestComputePetChunk:
    def test_uses_scalar_pet_evaluation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        df = pd.DataFrame(
            {
                "t": [10.0, 20.0],
                "mrt": [11.0, 21.0],
                "v": [1.0, 2.0],
                "rh": [50.0, 60.0],
            }
        )

        def fake_pet_corrected(
            tair: object,
            t_mrt: object,
            v_air: object,
            rh: object,
            *,
            icl: float,
        ) -> object:
            del t_mrt, v_air, rh, icl
            if not isinstance(tair, float):
                return 999.0
            return tair

        monkeypatch.setattr(calculate_pet, "pet_corrected", fake_pet_corrected)

        result = calculate_pet.compute_pet_chunk(df)

        assert result["pet"].tolist() == [10.0, 20.0]
