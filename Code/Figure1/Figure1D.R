library(dplyr)
library(lubridate)  # Process date and time variables

# Calculate gestational weeks using first-trimester start date as the reference date
clean_data_new <- clean_data %>%
  mutate(
    # Calculate the day difference between detection time and first-trimester start date
    day_difference = as.numeric(difftime(detection_time, first_trimester_start, units = "days")),
    # Divide the day difference by 7 to obtain rounded gestational weeks
    gestational_week = round(day_difference / 7, 0),
    # Define pregnancy stage according to gestational week range
    pregnancy_stage = case_when(
      gestational_week <= 13 ~ "First Trimester",
      gestational_week >= 14 & gestational_week <= 27 ~ "Second Trimester",
      gestational_week >= 28 ~ "Third Trimester",
      TRUE ~ NA_character_  # Mark uncomputable cases as NA
    )
  )

# Inspect the result
head(clean_data_new %>% select(detection_time, first_trimester_start, day_difference, gestational_week, pregnancy_stage))
# Save as CSV with row.names = FALSE to avoid row indices
write.csv(clean_data_new, "E:/pregnancy_gwas/pregnancy_stage/clean_data_new.csv", row.names = FALSE, fileEncoding = "UTF-8")



# Load required packages
library(ggplot2)
library(dplyr)

# Filter abnormal gestational weeks (1-42 weeks) and convert pregnancy stage to English
clean_data_filtered <- clean_data_new %>%
  filter(gestational_week > 0 & gestational_week <= 42) %>%
  mutate(
    trimester = case_when(
      pregnancy_stage == "First Trimester" ~ "First Trimester",
      pregnancy_stage == "Second Trimester" ~ "Second Trimester",
      pregnancy_stage == "Third Trimester" ~ "Third Trimester",
      TRUE ~ "Unknown"
    )
  )

# Create output directory if it does not exist
output_dir <- "E:/pregnancy_gwas/pregnancy_stage/blood_test_multi_plots"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Draw the English figure using the default sans font to avoid font errors
p <- ggplot(clean_data_filtered, aes(x = gestational_week, fill = trimester)) +
  geom_bar(color = "black", width = 0.8) +
  # Customize pregnancy-stage colors
  scale_fill_manual(
    values = c("First Trimester" = "#69b3a2", "Second Trimester" = "#404080", "Third Trimester" = "#f8766d"),
    na.value = "gray50"
  ) +
  # Use English labels to avoid encoding issues
  labs(
    title = "Distribution of Entries by Gestational Week",
    subtitle = "Data Range: 1-42 Weeks",
    x = "Gestational Week",
    y = "Number of Entries",
    fill = "Trimester"
  ) +
  # Set x-axis ticks every two weeks to avoid crowding
  scale_x_continuous(
    breaks = seq(1, 42, by = 2),
    limits = c(0.5, 42.5)
  ) +
  # Configure the theme using the default sans font
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.title.x = element_text(size = 14, margin = margin(t = 10)),
    axis.title.y = element_text(size = 14, margin = margin(r = 10)),
    axis.text.x = element_text(size = 10),
    axis.text.y = element_text(size = 10),
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 10),
    legend.position = "right"
  )

# Save as PDF using default settings
pdf_file <- file.path(output_dir, "gestational_week_distribution.pdf")
ggsave(
  filename = pdf_file,
  plot = p,
  device = "pdf",
  width = 10,
  height = 6,
  dpi = 300
)

# Print save-completion message
cat("Figure saved to: ", pdf_file, "\n")