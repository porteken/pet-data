DROP MATERIALIZED VIEW IF EXISTS public.pet_year_avg CASCADE;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year_max CASCADE;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year CASCADE;

DROP TABLE IF EXISTS public.pet_change;
DROP TABLE IF EXISTS public.pet_forecast;
DROP TABLE IF EXISTS public.pet_percentiles;
DROP TABLE IF EXISTS public.pet;
DROP TABLE IF EXISTS public.locations;

CREATE TABLE public.locations (
    location_id bigint PRIMARY KEY,
    city text NOT NULL,
    state text NOT NULL,
    lat double precision NOT NULL,
    lng double precision NOT NULL
);

CREATE TABLE public.pet (
    location_id bigint NOT NULL,
    date date NOT NULL,
    pet double precision NOT NULL
);

CREATE TABLE public.pet_percentiles (
    year integer NOT NULL,
    location_id bigint NOT NULL,
    p10 double precision NOT NULL,
    p90 double precision NOT NULL
);

CREATE TABLE public.pet_forecast (
    location_id bigint NOT NULL,
    year integer NOT NULL,
    pet double precision NOT NULL
);

CREATE TABLE public.pet_change (
    location_id bigint NOT NULL,
    decade text NOT NULL,
    change double precision NOT NULL
);
