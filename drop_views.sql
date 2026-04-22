set statement_timeout = '5s' ;
set lock_timeout = '1s' ;
DROP INDEX IF EXISTS pet_year_location_date_uidx ;
DROP INDEX IF EXISTS pet_year_location_year_idx ;
DROP INDEX IF EXISTS pet_year_year_idx ;
DROP INDEX IF EXISTS pet_year_avg_location_year_uidx ;
DROP INDEX IF EXISTS pet_year_avg_location_year_season_uidx ;
DROP INDEX IF EXISTS pet_year_avg_year_idx ;
DROP INDEX IF EXISTS pet_year_avg_season_idx ;
DROP INDEX IF EXISTS pet_year_max_location_year_uidx ;
DROP INDEX IF EXISTS pet_year_max_location_year_season_uidx ;
DROP INDEX IF EXISTS pet_year_max_year_idx ;
DROP INDEX IF EXISTS pet_year_max_season_idx ;
DROP INDEX IF EXISTS pet_percentiles_location_year_uidx ;
DROP INDEX IF EXISTS pet_percentiles_year_idx ;
DROP INDEX IF EXISTS pet_change_location_year_uidx ;
DROP INDEX IF EXISTS pet_change_year_idx ;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year_avg CASCADE ;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year_max ;
DROP MATERIALIZED VIEW IF EXISTS public.pet_year ;

DO $$
DECLARE
	relation_name text;
	relation_kind "char";
BEGIN
	FOREACH relation_name IN ARRAY ARRAY['pet_percentiles', 'pet_change'] LOOP
		SELECT c.relkind
		INTO relation_kind
		FROM pg_catalog.pg_class AS c
		JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
		WHERE n.nspname = 'public'
		AND c.relname = relation_name ;

		IF relation_kind = 'm' THEN
			EXECUTE format('DROP MATERIALIZED VIEW public.%I CASCADE', relation_name);
		ELSIF relation_kind IN ('r', 'p') THEN
			EXECUTE format('DROP TABLE public.%I CASCADE', relation_name);
		END IF;
	END LOOP;
END $$ ;
