# Previous package loading and data-cleaning code remains unchanged



# Generate groups in the order of factor levels within the loop
groups <- levels(test_data$pregnancy_stage)  # Extract factor levels directly because they are already ordered
comparisons <- combn(groups, 2, simplify = FALSE)  # Generate pairwise comparisons according to factor-level order
# Customize colors
custom_colors <- c(
  "First Trimester" = "#C9579F",    # Deep pink, matched to the example Normal group
  "Second Trimester" = "#57A6C9",    # Blue, matched to the example Tumor group
  "Third Trimester" = "#FF9E9E",
  "Puerperium" = "#FFCC99" # Light pink, matched to the example Metastatic group
)
# Loop over each test item to draw boxplots with pairwise P values
for (test in tests) {
  test_data <- clean_data %>%
    filter(test_item_name == test) %>%
    drop_na(pregnancy_stage, test_result)  # Ensure no NA values
  
  # Check the number of valid groups
  if (n_distinct(test_data$pregnancy_stage) < 2) {
    print(paste("Skipping", test, ": fewer than two valid groups"))
    next
  }
  
  # Extract groups by factor levels to ensure ordered pregnancy stages
  groups <- levels(test_data$pregnancy_stage)
  # Generate all pairwise group comparisons
  comparisons <- combn(groups, 2, simplify = FALSE)
  
  # Draw boxplots with pairwise intergroup P values
  p <- ggboxplot(
    test_data, 
    x = "pregnancy_stage", 
    y = "test_result", 
    fill = "pregnancy_stage",
    color = "pregnancy_stage",  # Match jitter point colors to groups
    palette = custom_colors,
    add = "jitter",       # Add jittered points
    alpha = 0.7,
    width = 0.6,
    size = 0.8            # Boxplot border width
  ) +
    # Add pairwise intergroup P values using Wilcoxon tests with Bonferroni correction
    stat_compare_means(
      method = "wilcox.test", 
      comparisons = comparisons,  # Specify pairwise comparison groups
      p.adjust.method = "bonferroni",  # Multiple-testing correction
      label = "p.signif",         # Show significance labels
      hide.ns = TRUE,             # Hide nonsignificant comparisons
      tip.length = 0.01,          # Tip length for comparison brackets
      size = 4                    # Significance label font size
    ) +
    theme_classic() +  # Use a clean theme matched to the target figure
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 12, family = "yahei"),
      axis.text.y = element_text(size = 12, family = "yahei"),
      plot.title = element_text(hjust = 0.5, size = 16, family = "yahei", face = "bold"),
      legend.position = "right",  # Legend position if shown
      legend.text = element_text(family = "yahei", size = 10),
      legend.title = element_text(family = "yahei", size = 11),
      axis.title.x = element_text(family = "yahei", size = 14),
      axis.title.y = element_text(family = "yahei", size = 14)
    ) +
    labs(
      title = paste("Test item: ", test),
      x = "pregnancy_stage",
      y = "test_result",
      fill = "pregnancy_stage"
    )
  
  # Save as PDF
  ggsave(
    filename = paste0(save_path, "/", test, ".pdf"),
    plot = p,
    device = cairo_pdf,
    width = 8,
    height = 6,
    dpi = 300
  )
  
  print(paste("Saved: ", test))
}

 showtext_auto(FALSE)
 
 
 
 
 
 
 for (test in tests) {
   test_data <- clean_data %>%
     filter(test_item_name == test) %>%
     drop_na(pregnancy_stage, test_result)
   
   if (n_distinct(test_data$pregnancy_stage) < 2) {
     print(paste("Skipping", test, ": fewer than two valid groups"))
     next
   }
   
   # Extract the maximum test result for vertical positioning
   max_val <- max(test_data$test_result, na.rm = TRUE)
   
   p <- ggboxplot(
     test_data, 
     x = "pregnancy_stage", 
     y = "test_result", 
     fill = "pregnancy_stage",
     color = "pregnancy_stage",  
     palette = custom_colors,
     add = "jitter",  
     alpha = 0.7,
     width = 0.6
   ) +
     # 1. First vs second trimester（Highest position）
     stat_compare_means(
       method = "wilcox.test",
       comparisons = list(c("First Trimester", "Second Trimester")),
       p.adjust.method = "bonferroni",
       label = "p.signif",
       hide.ns = TRUE,
       y.position = max_val * 1.2,  # Highest position
       size = 4
     ) +
     # 2. First vs third trimester（Second-highest position）
     stat_compare_means(
       method = "wilcox.test",
       comparisons = list(c("First Trimester", "Third Trimester")),
       p.adjust.method = "bonferroni",
       label = "p.signif",
       hide.ns = TRUE,
       y.position = max_val * 1.1,  # Second-highest position
       size = 4
     ) +
     # 3. Second vs third trimester（Lowest position）
     stat_compare_means(
       method = "wilcox.test",
       comparisons = list(c("Second Trimester", "Third Trimester")),
       p.adjust.method = "bonferroni",
       label = "p.signif",
       hide.ns = TRUE,
       y.position = max_val * 1.05,  # Lowest position
       size = 4
     ) +
     theme_classic() +
     theme(
       axis.text.x = element_text(angle = 0, hjust = 0.5, size = 12, family = "yahei"),
       axis.text.y = element_text(size = 12, family = "yahei"),
       plot.title = element_text(hjust = 0.5, size = 16, family = "yahei", face = "bold"),
       legend.text = element_text(family = "yahei", size = 10),
       legend.title = element_text(family = "yahei", size = 11),
       axis.title.x = element_text(family = "yahei", size = 14),
       axis.title.y = element_text(family = "yahei", size = 14)
     ) +
     labs(
       title = paste("Test item: ", test),
       x = "pregnancy_stage",
       y = "test_result",
       fill = "pregnancy_stage"
     )
   
   ggsave(
     filename = paste0(save_path, "/", test, ".pdf"),
     plot = p,
     device = cairo_pdf,
     width = 8,
     height = 6,
     dpi = 300
   )
   
   print(paste("Saved: ", test))
 }
 