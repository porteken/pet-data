set statement_timeout = '5s' ;
set lock_timeout = '1s' ;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year_avg CASCADE ;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year_max CASCADE ;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year CASCADE ;

-- Note: We use CREATE TABLE IF NOT EXISTS where possible, 
-- but for a full rebuild we might still want to DROP or just ALTER.
-- The user asked to "allow for any additional columns or changes", 
-- which implies ALTER but for simplicity in this pipeline, 
-- we can recreate or ensure columns exist.

CREATE TABLE IF NOT EXISTS public.locations (
location_id bigint PRIMARY KEY,
city text NOT NULL,
state text NOT NULL,
lat double precision NOT NULL,
lng double precision NOT NULL
) ;

CREATE TABLE IF NOT EXISTS public.pet (
location_id bigint NOT NULL,
date date NOT NULL,
pet double precision NOT NULL
) ;

CREATE INDEX if not exists id
ON public.pet (location_id, date) ;

CREATE TABLE IF NOT EXISTS public.pet_forecast (
location_id bigint NOT NULL,
year bigint NOT NULL,
pet double precision NOT NULL,
lower double precision,
upper double precision,
model_type text,
full_years_used bigint,
warming_rate double precision,
acceleration double precision
) ;

-- Handle migrations for pet_forecast if it already exists
DO $$ 
DECLARE
    forecast_table_name constant text := 'pet_forecast';
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name=forecast_table_name AND column_name='lower') THEN
        ALTER TABLE public.pet_forecast ADD COLUMN lower double precision;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name=forecast_table_name AND column_name='upper') THEN
        ALTER TABLE public.pet_forecast ADD COLUMN upper double precision;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name=forecast_table_name AND column_name='model_type') THEN
        ALTER TABLE public.pet_forecast ADD COLUMN model_type text;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name=forecast_table_name AND column_name='full_years_used') THEN
        ALTER TABLE public.pet_forecast ADD COLUMN full_years_used bigint;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name=forecast_table_name AND column_name='warming_rate') THEN
        ALTER TABLE public.pet_forecast ADD COLUMN warming_rate double precision;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name=forecast_table_name AND column_name='acceleration') THEN
        ALTER TABLE public.pet_forecast ADD COLUMN acceleration double precision;
    END IF;
END $$ ;

-- Derived analytics now live in create_views.sql as materialized views:
--   public.pet_percentiles
--   public.pet_change
