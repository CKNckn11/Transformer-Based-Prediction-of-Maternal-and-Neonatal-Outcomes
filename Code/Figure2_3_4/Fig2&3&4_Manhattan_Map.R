# ============================================================
#fig2,fig3,fig4 Manhattan Map
# ============================================================ 
library(data.table)
library(dplyr)
library(ggplot2)
library(ggrepel)
pheno <- "phenotype"

annot_file <- paste0("", pheno, "_annotated.annot")
col_names <- c("CHR", "SNP", "BP", "n_miss", "allele1", "allele0", "af", 
               "beta", "se", "logl_H1", "l_remle", "l_mle", "p_wald", "P", "ANNOT_RAW")
sumstat <- fread(annot_file, header = FALSE, skip = 1, col.names = col_names,
                 na.strings = c("", "NA", "."))
sumstat[, CHR := as.numeric(CHR)]
sumstat[, BP := as.numeric(BP)]
sumstat[, P := as.numeric(P)]
sumstat[, Gene := sub("^[0-9.eE+-]+\\s+", "", ANNOT_RAW)]
sumstat[, Gene := gsub("\\(.*\\)", "", Gene)]
sumstat[, Gene := gsub("[;,|].*", "", Gene)]
sumstat[grepl("^=", Gene), Gene := ""]
sumstat[Gene %in% c(".", "NONE", "missense"), Gene := ""]
sumstat[, Gene := trimws(Gene)]
filtered_data <- sumstat %>%
  mutate(CHR = as.numeric(CHR), BP = as.numeric(BP), P = as.numeric(P)) %>%
  filter(CHR %in% 1:23, P > 0, P <= 1, Gene != "", !is.na(Gene)) %>%
  arrange(CHR, BP)
if (nrow(filtered_data) == 0) stop("NO data")
sumstat_cum <- filtered_data %>%
  group_by(CHR) %>%
  summarise(chr_len = max(BP), .groups = "drop") %>%
  mutate(tot = cumsum(chr_len) - chr_len) %>%
  left_join(filtered_data, by = "CHR") %>%
  arrange(CHR, BP) %>%
  mutate(Positioncum = BP + tot)
axisdf <- sumstat_cum %>%
  group_by(CHR) %>%
  summarise(center = mean(range(Positioncum)), .groups = "drop") %>%
  mutate(label = ifelse(CHR == 23, "X", as.character(CHR)))
top10_genes <- sumstat_cum %>%
  ungroup() %>%
  filter(Gene != "") %>%
  group_by(Gene) %>%
  slice_min(order_by = P, n = 1) %>%
  ungroup() %>%
  arrange(P) %>%
  slice_head(n = 10) %>%
  distinct(Gene, .keep_all = TRUE)
sumstat_plot <- sumstat_cum %>%
  ungroup() %>%
  mutate(highlight = FALSE)
for (i in 1:nrow(top10_genes)) {
  this_chr <- top10_genes$CHR[i]
  this_bp  <- top10_genes$BP[i]
  sumstat_plot <- sumstat_plot %>%
    mutate(highlight = ifelse(CHR == this_chr &
                                BP >= this_bp - 500000 &
                                BP <= this_bp + 500000,
                              TRUE, highlight))
}
p <- ggplot(sumstat_plot, aes(x = Positioncum, y = -log10(P))) +
  geom_point(aes(color = as.factor(CHR)), alpha = 0.6, size = 0.8) +
  scale_color_manual(values = rep(c("#AAAAAA", "#DDDDDD"), 12)) +
  geom_point(data = subset(sumstat_plot, highlight), color = "red2", alpha = 0.9, size = 1) +
  geom_hline(yintercept = -log10(1e-6), color = "brown", linetype = "dashed", linewidth = 0.8) +
  scale_x_continuous(breaks = axisdf$center, labels = axisdf$label, expand = c(0.01, 0.01)) +
  scale_y_continuous(expand = c(0, 0.2)) +
  labs(title = paste0(pheno, " Manhattan Map"),
       x = " ", y = expression(-log[10](italic(P)))) +
  geom_label_repel(
    data = top10_genes,
    aes(label = Gene),
    color = "darkred",
    fill = "white",
    size = 5,               
    fontface = "bold.italic",   
    linewidth = 1,      
    segment.size = 1,
    box.padding = 1.6,    
    max.overlaps = 18,     
    show.legend = FALSE,   
    nudge_y = 3.5,          
    label.padding = unit(0.5, "lines"),  
    label.r = unit(0.6, "lines")         
  ) +
  theme_bw(base_size = 13) +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )
output_png <- paste0("", pheno, "_no_name.png")
png(output_png, units = "in", res = 300, height = 4, width = 12, pointsize = 8, bg = "white")
print(p)
dev.off()



