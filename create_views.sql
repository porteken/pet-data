CREATE MATERIALIZED VIEW public.pet_year_avg AS
SELECT
  p.location_id,
  EXTRACT(YEAR FROM p.date)::int AS year,
  ROUND(AVG(p.pet)::numeric, 1) AS pet
FROM public.pet AS p
GROUP BY
  p.location_id,
  EXTRACT(YEAR FROM p.date)::int;

CREATE UNIQUE INDEX pet_year_avg_location_year_uidx
  ON public.pet_year_avg (location_id, year);

CREATE INDEX pet_year_avg_year_idx
  ON public.pet_year_avg (year);

CREATE MATERIALIZED VIEW public.pet_year_max AS
SELECT
  p.location_id,
  EXTRACT(YEAR FROM p.date)::int AS year,
  ROUND(MAX(p.pet)::numeric, 1) AS pet
FROM public.pet AS p
GROUP BY
  p.location_id,
  EXTRACT(YEAR FROM p.date)::int;

CREATE UNIQUE INDEX pet_year_max_location_year_uidx
  ON public.pet_year_max (location_id, year);

CREATE INDEX pet_year_max_year_idx
  ON public.pet_year_max (year);

CREATE MATERIALIZED VIEW public.pet_year AS
SELECT
  p.location_id,
  p.date,
  EXTRACT(YEAR FROM p.date)::int AS year,
  p.pet
FROM public.pet AS p;

CREATE UNIQUE INDEX pet_year_location_date_uidx
  ON public.pet_year (location_id, date);

CREATE INDEX pet_year_location_year_idx
  ON public.pet_year (location_id, year);

CREATE INDEX pet_year_year_idx
  ON public.pet_year (year);
