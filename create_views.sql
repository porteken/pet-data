set statement_timeout = '5s' ;
set lock_timeout = '1s' ;
CREATE MATERIALIZED VIEW public.pet_year_avg AS
SELECT
p.location_id,
EXTRACT (YEAR FROM p.date)::int AS year,
s.season_name AS season,
ROUND (AVG (p.pet)::numeric, 1) AS pet
FROM public.pet AS p
JOIN (
VALUES
(1,
'Jan-Dec'),
(2,
'Jan-Dec'),
(3,
'Jan-Dec'),
(4,
'Jan-Dec'),
(5,
'Jan-Dec'),
(6,
'Jan-Dec'),
(7,
'Jan-Dec'),
(8,
'Jan-Dec'),
(9,
'Jan-Dec'),
(10,
'Jan-Dec'),
(11,
'Jan-Dec'),
(12,
'Jan-Dec'),
(2, 'Feb'),
(3, 'March-May'), (4, 'March-May'), (5, 'March-May'),
(6, 'June-August'), (7, 'June-August'), (8, 'June-August'),
(9,
'September-November'),
(10,
'September-November'),
(11,
'September-November')
) AS s (month, season_name)
ON EXTRACT (MONTH FROM p.date)::int = s.month
GROUP BY
p.location_id,
EXTRACT (YEAR FROM p.date)::int,
s.season_name ;

CREATE UNIQUE INDEX if not exists
pet_year_avg_location_year_season_uidx
ON public.pet_year_avg (location_id, year, season) ;

CREATE INDEX if not exists pet_year_avg_year_idx
ON public.pet_year_avg (year) ;

CREATE INDEX if not exists pet_year_avg_season_idx
ON public.pet_year_avg (season) ;

CREATE MATERIALIZED VIEW public.pet_year_max AS
SELECT
p.location_id,
EXTRACT (YEAR FROM p.date)::int AS year,
s.season_name AS season,
ROUND (MAX (p.pet)::numeric, 1) AS pet
FROM public.pet AS p
JOIN (
VALUES
(1,
'Jan-Dec'),
(2,
'Jan-Dec'),
(3,
'Jan-Dec'),
(4,
'Jan-Dec'),
(5,
'Jan-Dec'),
(6,
'Jan-Dec'),
(7,
'Jan-Dec'),
(8,
'Jan-Dec'),
(9,
'Jan-Dec'),
(10,
'Jan-Dec'),
(11,
'Jan-Dec'),
(12,
'Jan-Dec'),
(2, 'Feb'),
(3, 'March-May'), (4, 'March-May'), (5, 'March-May'),
(6, 'June-August'), (7, 'June-August'), (8, 'June-August'),
(9,
'September-November'),
(10,
'September-November'),
(11,
'September-November')
) AS s (month, season_name)
ON EXTRACT (MONTH FROM p.date)::int = s.month
GROUP BY
p.location_id,
EXTRACT (YEAR FROM p.date)::int,
s.season_name ;

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
