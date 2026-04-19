
C2K <- function(K) K - 273.15
SVP.Murray <- function(T) 6.1078 * exp(17.269 * T / (T + 237.3))
source("/home/kenneth-porter/pet_files/pet_corrected.R")
inputs <- read.csv("r_inputs_mini.csv")
results <- apply(inputs, 1, function(row) {
  PETcorrected(row["t"] - 273.15, row["v"], row["rh"], row["mrt"] - 273.15, icl=0.5)
})
write.csv(results, "r_results_mini.csv", row.names=FALSE)
