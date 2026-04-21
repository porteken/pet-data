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
