CREATE OR REPLACE VIEW public.pet_year_avg AS
SELECT
  pet.location_id,
  round(avg(pet.pet)::numeric, 1) as pet,
  date_part('year'::text, pet.date) as year
FROM
  pet
GROUP BY
  pet.location_id,
  (date_part('year'::text, pet.date));

CREATE OR REPLACE VIEW public.pet_year_max AS
SELECT
  pet.location_id,
  round(max(pet.pet)::numeric, 1) as pet,
  date_part('year'::text, pet.date) as year
FROM
  pet
GROUP BY
  pet.location_id,
  (date_part('year'::text, pet.date));

CREATE OR REPLACE VIEW public.pet_year AS
SELECT
  pet.location_id,
  pet.pet,
  pet.date,
  date_part('year'::text, pet.date) as year
FROM
  pet;