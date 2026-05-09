-- squawk-ignore-file require-concurrent-index-deletion
-- This file may be executed by tools that wrap the SQL in a transaction block,
-- so plain DROP INDEX statements are intentional here.

set statement_timeout = '5s' ;
set lock_timeout = '1s' ;
DROP INDEX IF EXISTS pet_year_stats_location_year_season_uidx ;
DROP INDEX IF EXISTS pet_year_stats_year_idx ;
DROP INDEX IF EXISTS pet_year_stats_season_idx ;
DROP INDEX IF EXISTS pet_forecast_location_year_season_uidx ;
DROP INDEX IF EXISTS pet_forecast_location_year_uidx ;
DROP INDEX IF EXISTS pet_forecast_year_idx ;
DROP INDEX IF EXISTS pet_forecast_season_idx ;
DROP INDEX IF EXISTS pet_change_location_year_uidx ;
DROP INDEX IF EXISTS pet_change_year_idx ;

DO $$
DECLARE
	relation_name text;
	relation_kind "char";
BEGIN
	FOREACH relation_name IN ARRAY ARRAY['pet_year_stats', 'city_rankings_view', 'pet_forecast', 'pet_change'] LOOP
		SELECT c.relkind
		INTO relation_kind
		FROM pg_catalog.pg_class AS c
		JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
		WHERE n.nspname = 'public'
		AND c.relname = relation_name ;

		IF relation_kind = 'm' THEN
			EXECUTE format('DROP MATERIALIZED VIEW public.%I CASCADE', relation_name);
		ELSIF relation_kind = 'v' THEN
			EXECUTE format('DROP VIEW public.%I CASCADE', relation_name);
		ELSIF relation_name <> 'pet_forecast' AND relation_kind IN ('r', 'p') THEN
			EXECUTE format('DROP TABLE public.%I CASCADE', relation_name);
		END IF;
	END LOOP;
END $$ ;
