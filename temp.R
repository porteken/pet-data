
source("/home/kenneth-porter/pet_files/pet_corrected.R")
res <- tryCatch({
  PETcorrected(25.3700008392334, 1.6299999952316284, 55.29999923706055, 36.95000076293945, icl=0.5)
}, error = function(e) {
  cat("ERROR:", conditionMessage(e))
  return(NA)
})
cat(res)
