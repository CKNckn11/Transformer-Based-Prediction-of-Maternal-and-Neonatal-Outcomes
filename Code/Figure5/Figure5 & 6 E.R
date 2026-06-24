library(ggplot2)

project_dir <- file.path(Sys.getenv("USERPROFILE"), "Desktop", "\u5b55\u5987\u6587\u7ae0")
input_file <- file.path(project_dir, "model1_snp_top10_mean_shap_\u6574\u5408.csv")
output_dir <- file.path(project_dir, "model1_shap_lollipop")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

data <- read.csv(input_file, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)
data$Mean_SHAP_Value <- as.numeric(data$Mean_SHAP_Value)

sanitize_filename <- function(x) {
  x <- gsub("[\\\\/:*?\"<>|]", "_", x)
  x <- gsub("\\s+", "_", x)
  x
}

plot_one_disease <- function(df, phenotype) {
  df <- df[order(df$Rank), ]
  df$Feature_Label <- gsub("^SNP_", "", df$Feature_Name)
  df$Feature_Label <- factor(df$Feature_Label, levels = df$Feature_Label)
  df$Direction <- ifelse(df$Mean_SHAP_Value >= 0, "Positive", "Negative")

  y_min <- min(df$Mean_SHAP_Value, 0, na.rm = TRUE)
  y_max <- max(df$Mean_SHAP_Value, 0, na.rm = TRUE)
  y_pad <- max((y_max - y_min) * 0.18, 0.01)

  ggplot(df, aes(x = Feature_Label, y = Mean_SHAP_Value, color = Direction)) +
    geom_hline(yintercept = 0, color = "#9a9a9a", linewidth = 0.8, linetype = "dashed") +
    geom_segment(aes(xend = Feature_Label, y = 0, yend = Mean_SHAP_Value), linewidth = 1.1) +
    geom_point(size = 3.6) +
    scale_color_manual(values = c("Positive" = "#E15759", "Negative" = "#4E89A8")) +
    scale_y_continuous(limits = c(y_min - y_pad, y_max + y_pad), expand = c(0.02, 0.02)) +
    labs(title = phenotype, x = NULL, y = "Mean SHAP Value") +
    theme_bw(base_size = 10) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
      panel.border = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line.x = element_blank(),
      axis.line.y = element_blank(),
      axis.ticks.x = element_blank(),
      axis.text.x = element_text(angle = 50, hjust = 1, size = 8),
      axis.text.y = element_text(size = 8),
      axis.title.y = element_text(size = 9),
      legend.position = "none"
    )
}

phenotypes <- unique(data$Phenotype)

for (phenotype in phenotypes) {
  disease_data <- data[data$Phenotype == phenotype, ]
  p <- plot_one_disease(disease_data, phenotype)
  base_name <- sanitize_filename(phenotype)

  ggsave(
    file.path(output_dir, paste0(base_name, "_snp_top10_mean_shap_lollipop.pdf")),
    p,
    width = 5.2,
    height = 3.8
  )
  ggsave(
    file.path(output_dir, paste0(base_name, "_snp_top10_mean_shap_lollipop.png")),
    p,
    width = 5.2,
    height = 3.8,
    dpi = 300
  )
}

cat("Phenotypes:", length(phenotypes), "\n")
cat("Saved to:", output_dir, "\n")
