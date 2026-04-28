"""Utility for computing PET data date ranges."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone

from shared_config import mrt_available_end_date


def main() -> None:
    """Output year ranges for PET data processing."""
    now = datetime.now(tz=timezone.utc)
    prev_year = now.year - 1

    era5_start = 2000

    arco_safe_month = 4
    era5_end = now.year - 2 if now.month < arco_safe_month else prev_year

    era5_end = min(era5_end, prev_year)

    era5_years = list(range(era5_start, era5_end + 1))

    cds_start = era5_end + 1
    end_date = mrt_available_end_date()
    end_date = min(end_date, date(prev_year, 12, 31))

    cds_years = []
    cds_year_months = []
    if cds_start <= end_date.year:
        for year in range(cds_start, end_date.year + 1):
            cds_years.append(year)
            m_end = 12 if year < end_date.year else end_date.month
            cds_year_months.extend(
                {
                    "year": year,
                    "month": month,
                    "month_pad": f"{month:02d}",
                }
                for month in range(1, m_end + 1)
            )

    years = sorted(set(era5_years + cds_years))

    if "--shell" in sys.argv:
        print(f"ERA5_YEARS='{' '.join(map(str, era5_years))}'")
        print(f"CDS_YEARS='{' '.join(map(str, cds_years))}'")
        print(f"CDS_YEAR_MONTHS='{json.dumps(cds_year_months)}'")
        print(f"ALL_YEARS='{' '.join(map(str, years))}'")
        print(f"END_DATE='{end_date.isoformat()}'")
        print(f"START_YEAR='{era5_start}'")
        print(f"END_YEAR='{end_date.year}'")

    elif "--github" in sys.argv:
        if "--yearly" in sys.argv:
            target_year = prev_year
            print(f"target_year={target_year}")
            print(f"window_start={target_year}-01-01")
            print(f"window_end={target_year}-12-31")
            print(f"months={json.dumps(list(range(1, 13)))}")
            print(f"analytics_shards={json.dumps(list(range(20)))}")
        else:
            print(f"start_year={era5_start}")
            print(f"end_year={end_date.year}")
            print(f"end_date={end_date.isoformat()}")
            print(f"years={json.dumps(years)}")
            print(f"era5_years={json.dumps(era5_years)}")
            print(f"cds_years={json.dumps(cds_years)}")
            print(f"cds_year_months={json.dumps(cds_year_months)}")
            print(f"analytics_shards={json.dumps(list(range(20)))}")


if __name__ == "__main__":
    main()
