# ============================================================
# Extended Data Table 1
# ============================================================ 
library(grid)
n_samples      <- 
  n_total        <- 
  n_snps         <- 
  n_indels       <- 
  n_dbsnp_known  <- 
  n_dbsnp_absent <- 
  n_pass         <- 
  tstv           <- 
  pct_absent <- n_dbsnp_absent / n_total * 100   # 59.3%
pct_pass  <- n_pass / n_total * 100             # 87.3%
fmt_int <- function(x) format(as.integer(x), big.mark = ",", scientific = FALSE)
fmt_pct <- function(x) sprintf("%.1f%%", x)
rows <- list()
rows[[length(rows) + 1]] <- list(cat = "Sequencing overview",          val = "",                section = TRUE)
rows[[length(rows) + 1]] <- list(cat = "Samples sequenced",            val = fmt_int(n_samples), section = FALSE)
rows[[length(rows) + 1]] <- list(cat = "Variant discovery",            val = "",                section = TRUE)
rows[[length(rows) + 1]] <- list(cat = "Total variants",               val = fmt_int(n_total),  section = FALSE)
rows[[length(rows) + 1]] <- list(cat = "  SNPs",                       val = fmt_int(n_snps),   section = FALSE)
rows[[length(rows) + 1]] <- list(cat = "  Indels",                     val = fmt_int(n_indels), section = FALSE)
rows[[length(rows) + 1]] <- list(cat = "Variant annotation",           val = "",                section = TRUE)
rows[[length(rows) + 1]] <- list(cat = "Variants annotated in dbSNP138", val = fmt_int(n_dbsnp_known), section = FALSE)
rows[[length(rows) + 1]] <- list(cat = "Variants absent from dbSNP138", val = fmt_int(n_dbsnp_absent), section = FALSE)
rows[[length(rows) + 1]] <- list(cat = "  Proportion of total absent",  val = fmt_pct(pct_absent),     section = FALSE)
rows[[length(rows) + 1]] <- list(cat = "Quality metrics",              val = "",                section = TRUE)
rows[[length(rows) + 1]] <- list(cat = "PASS variants",               val = fmt_int(n_pass),   section = FALSE)
rows[[length(rows) + 1]] <- list(cat = "  PASS proportion",           val = fmt_pct(pct_pass), section = FALSE)
rows[[length(rows) + 1]] <- list(cat = "Ts/Tv ratio",                val = sprintf("%.2f", tstv), section = FALSE)
draw_edt1_table <- function(
    rows,
    title_text,
    footnotes,
    out_pdf,
    font_size     = 9,
    header_size   = 9.5,
    section_size  = 9,
    title_size    = 11,
    footnote_size = 7.5,
    page_width    = 8.27,  
    page_height   = 11.69    
) {
  nr <- length(rows)
  margin_left_inch   <- 0.75
  margin_right_inch  <- 0.75
  margin_top_inch    <- 0.65
  margin_bottom_inch <- 0.60
  margin_left  <- margin_left_inch  / page_width          
  margin_right <- 1 - margin_right_inch / page_width      
  table_left  <- margin_left
  table_right <- margin_right
  table_w     <- table_right - table_left                
  col1_frac  <- 0.60
  col1_left  <- table_left
  col1_right <- table_left + table_w * col1_frac
  col2_right <- table_right
  pad <- 0.006   
  title_h      <- 0.04
  header_h     <- 0.032
  normal_row_h <- 0.026
  section_h    <- 0.032
  table_top <- 1 - margin_top_inch / page_height - title_h
  n_section    <- sum(sapply(rows, function(r) r$section))
  n_data       <- nr - n_section
  total_needed <- header_h + n_section * section_h + n_data * normal_row_h + 0.04
  footnote_block_h <- 0.06 + length(footnotes) * 0.016
  available <- table_top - margin_bottom_inch / page_height - footnote_block_h}
if (total_needed > available) {
  scale_f      <- available / total_needed
  normal_row_h <- normal_row_h * scale_f
  section_h    <- section_h    * scale_f
  header_h     <- header_h     * min(scale_f, 1.0)
  message("Row heights auto-scaled (scale=", round(scale_f, 3), ")")
}
cairo_pdf(file = out_pdf, width = page_width, height = page_height, family = "Arial")
grid.newpage()
title_y <- 1 - margin_top_inch / page_height
grid.text(
  title_text,
  x = unit(margin_left, "npc"), y = unit(title_y, "npc"),
  just = c("left", "top"),
  gp = gpar(fontsize = title_size, fontface = "bold", col = "black")
)
header_y_mid <- table_top - header_h / 2
grid.rect(
  x = unit((table_left + table_right) / 2, "npc"), y = unit(header_y_mid, "npc"),
  width = unit(table_w, "npc"), height = unit(header_h, "npc"),
  gp = gpar(fill = "#D9D8C8", col = NA)
)
grid.text("Category",
          x = unit(col1_left + pad, "npc"), y = unit(header_y_mid, "npc"),
          just = c("left", "center"),
          gp = gpar(fontsize = header_size, fontface = "bold", col = "black")
)
grid.text("Value",
          x = unit(col2_right - pad, "npc"), y = unit(header_y_mid, "npc"),
          just = c("right", "center"),
          gp = gpar(fontsize = header_size, fontface = "bold", col = "black")
)
grid.lines(x = unit(c(table_left, table_right), "npc"),
           y = unit(c(table_top, table_top), "npc"),
           gp = gpar(lwd = 1.5, col = "black"))
y_header_bottom <- table_top - header_h
grid.lines(x = unit(c(table_left, table_right), "npc"),
           y = unit(c(y_header_bottom, y_header_bottom), "npc"),
           gp = gpar(lwd = 0.8, col = "black"))
current_y <- y_header_bottom
data_seq  <- 0
for (i in seq_len(nr)) {
  r      <- rows[[i]]
  is_sec <- r$section
  row_h  <- if (is_sec) section_h else normal_row_h
  y_mid  <- current_y - row_h / 2
  y_bot  <- current_y - row_h
  if (is_sec) {
    grid.rect(
      x = unit((table_left + table_right) / 2, "npc"), y = unit(y_mid, "npc"),
      width = unit(table_w, "npc"), height = unit(row_h, "npc"),
      gp = gpar(fill = "#E8E6DD", col = NA)
    )
    grid.text(r$cat,
              x = unit(col1_left + pad, "npc"), y = unit(y_mid, "npc"),
              just = c("left", "center"),
              gp = gpar(fontsize = section_size, fontface = "bold", col = "black")
    )
    grid.lines(x = unit(c(table_left, table_right), "npc"),
               y = unit(c(y_bot, y_bot), "npc"),
               gp = gpar(lwd = 0.8, col = "black"))
  } else {
    data_seq <- data_seq + 1
    if (data_seq %% 2 == 0) {
      grid.rect(
        x = unit((table_left + table_right) / 2, "npc"), y = unit(y_mid, "npc"),
        width = unit(table_w, "npc"), height = unit(row_h, "npc"),
        gp = gpar(fill = "#F4F4EF", col = NA)
      )
    }
    grid.text(r$cat,
              x = unit(col1_left + pad, "npc"), y = unit(y_mid, "npc"),
              just = c("left", "center"),
              gp = gpar(fontsize = font_size, col = "black")
    )
    grid.text(r$val,
              x = unit(col2_right - pad, "npc"), y = unit(y_mid, "npc"),
              just = c("right", "center"),
              gp = gpar(fontsize = font_size, col = "black")
    )
  }
  current_y <- y_bot
}
grid.lines(x = unit(c(table_left, table_right), "npc"),
           y = unit(c(current_y, current_y), "npc"),
           gp = gpar(lwd = 1.5, col = "black"))
fn_y0 <- current_y - 0.025
for (fi in seq_along(footnotes)) {
  grid.text(footnotes[fi],
            x = unit(margin_left, "npc"), y = unit(fn_y0 - (fi - 1) * 0.016, "npc"),
            just = c("left", "top"),
            gp = gpar(fontsize = footnote_size, col = "#444444")
  )
}
dev.off()
title_text <- paste0()

draw_edt1_table(
  rows       = rows,
  title_text = title_text,
  footnotes  = footnotes,
  out_pdf    = OUT_PDF
)


