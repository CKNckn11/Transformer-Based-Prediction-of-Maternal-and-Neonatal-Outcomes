library(ggplot2)

project_dir <- file.path(Sys.getenv("USERPROFILE"), "Desktop", "\u5b55\u5987\u6587\u7ae0")
input_file <- file.path(project_dir, "model2_\u4ea4\u53c9\u9a8c\u8bc1\u6307\u6807\u6c47\u603b_\u6574\u5408.csv")
output_pdf <- file.path(project_dir, "Figure6_D_model2_matched.pdf")
output_png <- file.path(project_dir, "Figure6_D_model2_matched.png")

metrics <- read.csv(input_file, fileEncoding = "GBK", stringsAsFactors = FALSE)

mean_rows <- metrics[grepl("Mean", metrics[[4]]), ]

labels_by_file_order <- c(
  "Low birth weight",
  "GDM",
  "PIH",
  "ASD",
  "SCH",
  "Bilirubin"
)
mean_rows$Label <- labels_by_file_order[seq_len(nrow(mean_rows))]

plot_data <- rbind(
  data.frame(Label = mean_rows$Label, Panel = "NPV", Mean = suppressWarnings(as.numeric(mean_rows[[15]]))),
  data.frame(Label = mean_rows$Label, Panel = "PPV", Mean = suppressWarnings(as.numeric(mean_rows[[14]]))),
  data.frame(Label = mean_rows$Label, Panel = "Sensitivity", Mean = suppressWarnings(as.numeric(mean_rows[[10]]))),
  data.frame(Label = mean_rows$Label, Panel = "Specificity", Mean = suppressWarnings(as.numeric(mean_rows[[11]])))
)

plot_data$Label <- factor(plot_data$Label, levels = c("GDM", "PIH", "SCH", "Bilirubin", "Low birth weight", "ASD"))
plot_data$Panel <- factor(plot_data$Panel, levels = c("NPV", "PPV", "Sensitivity", "Specificity"))

point_colors <- c(
  "GDM" = "#4C78A8",
  "PIH" = "#E45756",
  "SCH" = "#54A24B",
  "Bilirubin" = "#F58518",
  "Low birth weight" = "#B279A2",
  "ASD" = "#72B7B2"
)

p <- ggplot(plot_data, aes(x = Label, y = Mean, color = Label)) +
  geom_point(size = 3.0, na.rm = TRUE) +
  facet_wrap(~Panel, ncol = 2) +
  scale_color_manual(values = point_colors) +
  scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.3, 0.6, 0.9), expand = c(0.02, 0.02)) +
  labs(
    title = "Distribution of Diagnostic Metrics",
    x = NULL,
    y = "Metric Value"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    strip.background = element_rect(fill = "#cfcfcf", color = "black", linewidth = 0.8),
    strip.text = element_text(face = "bold", size = 8),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
    axis.text.y = element_text(size = 9),
    axis.title.y = element_text(size = 11),
    legend.position = "none"
  )

ggsave(output_pdf, p, width = 7.8, height = 5.3)
ggsave(output_png, p, width = 7.8, height = 5.3, dpi = 300)

cat("Saved PDF to:", output_pdf, "\n")
cat("Saved PNG to:", output_png, "\n")
