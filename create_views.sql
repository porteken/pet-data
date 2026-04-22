set statement_timeout = '0' ;
set lock_timeout = '1s' ;
CREATE MATERIALIZED VIEW public.pet_year_avg AS
WITH pet_with_seasons AS (
SELECT
p.location_id,
EXTRACT (YEAR FROM p.date)::int AS year,
'Annual'::text AS season,
p.pet
FROM public.pet AS p
UNION ALL
SELECT
p.location_id,
EXTRACT (YEAR FROM p.date)::int AS year,
CASE
WHEN EXTRACT (MONTH FROM p.date)::int IN (12, 1, 2) THEN 'Winter'
WHEN EXTRACT (MONTH FROM p.date)::int IN (3, 4, 5) THEN 'Spring'
WHEN EXTRACT (MONTH FROM p.date)::int IN (6, 7, 8) THEN 'Summer'
ELSE 'Fall'
END AS season,
p.pet
FROM public.pet AS p
)
SELECT
location_id,
year,
season,
ROUND (AVG (pet)::numeric, 1) AS pet
FROM pet_with_seasons
GROUP BY
location_id,
year,
season ;

CREATE UNIQUE INDEX if not exists
pet_year_avg_location_year_season_uidx
ON public.pet_year_avg (location_id, year, season) ;

CREATE INDEX if not exists pet_year_avg_year_idx
ON public.pet_year_avg (year) ;

CREATE INDEX if not exists pet_year_avg_season_idx
ON public.pet_year_avg (season) ;

CREATE MATERIALIZED VIEW public.pet_year_max AS
WITH pet_with_seasons AS (
SELECT
p.location_id,
EXTRACT (YEAR FROM p.date)::int AS year,
'Annual'::text AS season,
p.pet
FROM public.pet AS p
UNION ALL
SELECT
p.location_id,
EXTRACT (YEAR FROM p.date)::int AS year,
CASE
WHEN EXTRACT (MONTH FROM p.date)::int IN (12, 1, 2) THEN 'Winter'
WHEN EXTRACT (MONTH FROM p.date)::int IN (3, 4, 5) THEN 'Spring'
WHEN EXTRACT (MONTH FROM p.date)::int IN (6, 7, 8) THEN 'Summer'
ELSE 'Fall'
END AS season,
p.pet
FROM public.pet AS p
)
SELECT
location_id,
year,
season,
ROUND (MAX (pet)::numeric, 1) AS pet
FROM pet_with_seasons
GROUP BY
location_id,
year,
season ;

CREATE UNIQUE INDEX if not exists
pet_year_max_location_year_season_uidx
ON public.pet_year_max (location_id, year, season) ;

CREATE INDEX if not exists pet_year_max_year_idx
ON public.pet_year_max (year) ;

CREATE INDEX if not exists pet_year_max_season_idx
ON public.pet_year_max (season) ;

CREATE MATERIALIZED VIEW public.pet_year AS
SELECT
p.location_id,
p.date,
EXTRACT (YEAR FROM p.date)::int AS year,
p.pet
FROM public.pet AS p ;

CREATE UNIQUE INDEX if not exists pet_year_location_date_uidx
ON public.pet_year (location_id, date) ;

CREATE INDEX if not exists pet_year_location_year_idx
ON public.pet_year (location_id, year) ;

CREATE INDEX if not exists pet_year_year_idx
ON public.pet_year (year) ;

CREATE MATERIALIZED VIEW public.pet_percentiles AS
WITH pet_with_year AS (
SELECT
EXTRACT (YEAR FROM p.date)::int AS year,
p.location_id,
p.pet
FROM public.pet AS p
)
SELECT
year,
location_id,
ROUND ((PERCENTILE_CONT (0.1) WITHIN GROUP (ORDER BY pet))::numeric, 1) AS p10,
ROUND ((PERCENTILE_CONT (0.9) WITHIN GROUP (ORDER BY pet))::numeric, 1) AS p90
FROM pet_with_year
GROUP BY
year,
location_id ;

CREATE UNIQUE INDEX if not exists pet_percentiles_location_year_uidx
ON public.pet_percentiles (location_id, year) ;

CREATE INDEX if not exists pet_percentiles_year_idx
ON public.pet_percentiles (year) ;

CREATE MATERIALIZED VIEW public.pet_change AS
WITH daily_pet AS (
SELECT
p.location_id,
p.date,
AVG (p.pet) AS pet
FROM public.pet AS p
GROUP BY
p.location_id,
p.date
), historical_yearly_avg AS (
SELECT
d.location_id,
EXTRACT (YEAR FROM d.date)::int AS year,
AVG (d.pet) AS pet,
0 AS source_order
FROM daily_pet AS d
GROUP BY
d.location_id,
EXTRACT (YEAR FROM d.date)::int
), combined_yearly_avg AS (
SELECT
location_id,
year,
pet,
source_order
FROM historical_yearly_avg
UNION ALL
SELECT
f.location_id,
f.year::int AS year,
f.pet,
1 AS source_order
FROM public.pet_forecast AS f
), deduplicated_yearly_avg AS (
SELECT DISTINCT ON (location_id, year)
location_id,
year,
pet
FROM combined_yearly_avg
ORDER BY
location_id,
year,
source_order
), decade_avg AS (
SELECT
location_id,
(year / 10) * 10 AS year,
AVG (pet) AS pet
FROM deduplicated_yearly_avg
GROUP BY
location_id,
(year / 10) * 10
), change_rows AS (
SELECT
location_id,
year,
ROUND ((pet - LAG (pet) OVER (PARTITION BY location_id ORDER BY year))::numeric,
2) AS change
FROM decade_avg
)
SELECT
location_id,
year,
change
FROM change_rows
WHERE change IS NOT NULL ;

CREATE UNIQUE INDEX if not exists pet_change_location_year_uidx
ON public.pet_change (location_id, year) ;

CREATE INDEX if not exists pet_change_year_idx
ON public.pet_change (year) ;
