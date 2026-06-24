# ============================================================
# QQ plots
# ============================================================ 
library(dplyr)
library(CMplot)
library(colorspace)
library(grDevices)
output_dir <- "QQplot"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}
assoc_files <- list.files(
  pattern = "_gwas.assoc.txt$",
  full.names = TRUE
)
calc_lambda_gc <- function(p) {
  chisq <- qchisq(1 - p, 1)
  lambda_gc <- median(chisq, na.rm = TRUE) /
    qchisq(0.5, 1)
  return(lambda_gc)
}
for (assoc_file in assoc_files) {
  cat("Processing:", assoc_file, "\n")
  phenotype <- basename(assoc_file)
  phenotype <- sub(
    "_gwas.assoc.txt",
    "",
    phenotype
  )
  data <- read.table(
    assoc_file,
    header = TRUE,
    stringsAsFactors = FALSE
  )
  colnames(data)[colnames(data) == "p_lrt"] <- "P"
  colnames(data)[colnames(data) == "rs"] <- "SNP"
  colnames(data)[colnames(data) == "chr"] <- "CHR"
  colnames(data)[colnames(data) == "ps"] <- "BP"
  manhattan_data <- data %>%
    select(SNP, CHR, BP, P)
  manhattan_data <- manhattan_data %>%
    filter(
      !is.na(P),
      is.finite(P),
      P > 0,
      P <= 1
    )
  lambda_gc <- calc_lambda_gc(
    manhattan_data$P
  )
  png(
    filename = file.path(
      output_dir,
      paste0(phenotype, "_QQ.png")
    ),
    width = 3000,
    height = 3000,
    res = 600
  )
  CMplot( manhattan_data, 
          plot.type = "q",
          col = "dodgerblue1",
          pch = 19, cex = 0.6,
          conf.int = TRUE,
          box = FALSE,
          threshold = 1e-6,
          threshold.col = "red",
          threshold.lty = 2,
          axis.cex = 1.5,
          lab.cex = 1.5,
          ylab = "",
          ylab.pos = 2.5,
          main = "",
          file.output = FALSE,
          verbose = TRUE, 
  )
  legend(
    "topleft",
    legend = bquote(
      lambda[GC] == .(
        round(lambda_gc, 3)
      )
    ),
    bty = "n",
    cex = 1.3
  )
  dev.off()
  cat(
    "Finished:",
    phenotype,
    "| lambda GC =",
    round(lambda_gc, 3),
    "\n\n"
  )
}
cat("All QQ plots completed!\n")




