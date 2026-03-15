DROP INDEX IF EXISTS pet_year_location_date_uidx;
DROP INDEX IF EXISTS pet_year_location_year_idx;
DROP INDEX IF EXISTS pet_year_year_idx;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year_avg;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year_max;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year;
