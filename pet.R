suppressPackageStartupMessages({
  library(arrow)
  library(tidyverse)
  library(doSNOW)
  library(foreach)
})


message("1. Loading and preprocessing data...")


df <- read_parquet("combined_data.parquet") %>%
  select(location_id, time, v, t, rh, mrt) %>%
  filter(rh >= 1 & v > 0) %>%
  mutate(across(c(v, t, rh, mrt), ~ round(.x * 2) / 2))

message("2. Extracting distinct weather combinations...")

df_distinct <- df %>%
  select(v, t, rh, mrt) %>%
  distinct()

dfs <- split(df_distinct, ceiling(seq_len(nrow(df_distinct)) / 1000))

message(sprintf(
  "3. Starting parallel computation for %d unique combinations...",
  nrow(df_distinct)
))

n_cores <- parallel::detectCores() - 1
my_cluster <- makeCluster(n_cores)
registerDoSNOW(cl = my_cluster)

pb <- txtProgressBar(max = length(dfs), style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

source("pet_corrected.R")

start_time <- Sys.time()

df_pet_unique <- foreach(
  d = dfs,
  .combine = bind_rows,
  .options.snow = opts,
  .packages = c("dplyr", "humidity"),
  .export = c("PETcorrected")
) %dopar%
  {
    d %>%
      rowwise() %>%
      mutate(pet = PETcorrected(t, mrt, v, rh, icl = 0.5)) %>%
      ungroup() %>%
      mutate(pet = round(pet * 2) / 2)
  }

close(pb)
stopCluster(my_cluster)

message(sprintf(
  "\nComputation finished in %s",
  format(Sys.time() - start_time)
))


message("4. Joining calculated PET back to main dataset...")

df_joined <- df %>%
  inner_join(df_pet_unique, by = c("v", "t", "rh", "mrt"))

message("5. Aggregating daily max PET per location...")


df_final <- df_joined %>%
  mutate(date = as.Date(time)) %>%
  group_by(location_id, date) %>%
  summarise(pet = max(pet, na.rm = TRUE), .groups = "drop") %>%

  mutate(id = row_number())

message("6. Saving final outputs...")
write.csv(df_final, "pet.csv", row.names = FALSE)

message("Pipeline Complete!")
