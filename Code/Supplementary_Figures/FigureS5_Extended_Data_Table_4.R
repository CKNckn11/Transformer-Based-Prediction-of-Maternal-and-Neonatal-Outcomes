# ============================================================
# Extended Data Table 4
# ============================================================ 
library(dplyr)
library(tidyr)
library(grid)
library(readxl)
raw <- readxl::read_excel(xlsx_path, sheet = 1, col_names = TRUE)
colnames(raw) <- c("Category", "Phenotype", "ICD10")
raw$ICD10_display <- gsub("\\n", ", ", raw$ICD10, fixed = FALSE)
draw_df <- data.frame(
  Phenotype = character(),
  ICD10     = character(),
  is_header = logical(),
  stringsAsFactors = FALSE
)
current_cat <- ""
for (i in seq_len(nrow(raw))) {
  this_cat <- as.character(raw$Category[i])
  if (this_cat != current_cat) {
    draw_df <- rbind(draw_df, data.frame(
      Phenotype = paste0(toupper(substring(this_cat, 1, 1)),
                         substring(this_cat, 2), " Phenotypes"),
      ICD10     = "",
      is_header = TRUE,
      stringsAsFactors = FALSE
    ))
    current_cat <- this_cat
  }
  draw_df <- rbind(draw_df, data.frame(
    Phenotype = as.character(raw$Phenotype[i]),
    ICD10     = as.character(raw$ICD10_display[i]),
    is_header = FALSE,
    stringsAsFactors = FALSE
  ))
}
draw_icd10_table <- function(
    data,
    font_size   = 7.5,
    header_size = 8,
    title_size  = 10,
    page_width  = 8.27,
    page_height = 11.69
) {
  nr <- nrow(data)
  nc <- ncol(data)
  nd <- 2
  margin_left  <- 0.10                     
  margin_right <- 0.10                     
  usable_w     <- 1 - margin_left - margin_right   
  name_rel  <- 0.28
  icd_rel   <- 0.72
  col_widths   <- c(name_rel * usable_w,
                    icd_rel  * usable_w)
  col_x_left   <- numeric(nd)
  col_x_center <- numeric(nd)
  cum <- margin_left
  for (j in seq_len(nd)) {
    col_x_left[j]   <- cum
    col_x_center[j] <- cum + col_widths[j] / 2
    cum <- cum + col_widths[j]
  }
  table_right <- cum                      
  title_h     <- 0.050
  header_h    <- 0.035
  normal_row_h<- 0.022
  section_h   <- 0.028
  table_top   <- 0.96 - title_h         
  n_section   <- sum(data$is_header, na.rm = TRUE)
  n_data      <- nr - n_section
  total_needed <- header_h + n_section * section_h + n_data * normal_row_h + 0.06
  available   <- table_top - 0.04
  if (total_needed > available) {
    scale_f    <- available / total_needed
    normal_row_h <- normal_row_h * scale_f
    section_h    <- section_h * scale_f
    header_h     <- header_h * min(scale_f, 1.0)
  }
  pdf(file = out_pdf, width = page_width, height = page_height)
  grid.newpage()
  grid.text(
    title_text,
    x    = unit(margin_left, "npc"),
    y    = unit(0.97, "npc"),
    just = c("left", "top"),
    gp   = gpar(fontsize = title_size, fontface = "bold", col = "black")
  )
  header_y_mid <- table_top - header_h / 2
  grid.rect(
    x      = unit((margin_left + table_right) / 2, "npc"),
    y      = unit(header_y_mid, "npc"),
    width  = unit(table_right - margin_left, "npc"),
    height = unit(header_h, "npc"),
    gp     = gpar(fill = "#D9D8C8", col = NA)
  )
  col_labels <- c("Phenotype", "ICD-10 Code(s)")
  for (j in seq_len(nd)) {
    grid.text(
      col_labels[j],
      x    = unit(col_x_center[j], "npc"),
      y    = unit(header_y_mid, "npc"),
      just = c("center", "center"),
      gp   = gpar(fontsize = header_size, fontface = "bold", col = "black")
    )
  }
  grid.lines(
    x  = unit(c(margin_left, table_right), "npc"),
    y  = unit(c(table_top, table_top), "npc"),
    gp = gpar(lwd = 1.2, col = "black")
  )
  y_header_bottom <- table_top - header_h
  grid.lines(
    x  = unit(c(margin_left, table_right), "npc"),
    y  = unit(c(y_header_bottom, y_header_bottom), "npc"),
    gp = gpar(lwd = 0.7, col = "black")
  )
  current_y <- y_header_bottom
  for (i in seq_len(nr)) {
    is_section <- data$is_header[i]
    row_h  <- if (is_section) section_h else normal_row_h
    y_mid <- current_y - row_h / 2
    y_bot <- current_y - row_h
    if (is_section) {
      grid.rect(
        x      = unit((margin_left + table_right) / 2, "npc"),
        y      = unit(y_mid, "npc"),
        width  = unit(table_right - margin_left, "npc"),
        height = unit(row_h, "npc"),
        gp     = gpar(fill = "#E8E6DD", col = NA)
      )
      grid.text(
        data$Phenotype[i],
        x    = unit(margin_left + 0.005, "npc"),
        y    = unit(y_mid, "npc"),
        just = c("left", "center"),
        gp   = gpar(fontsize = font_size + 0.5, fontface = "bold", col = "black")
      )
      grid.lines(
        x  = unit(c(margin_left, table_right), "npc"),
        y  = unit(c(y_bot, y_bot), "npc"),
        gp = gpar(lwd = 0.7, col = "black")
      )
    } else {
      data_seq_i <- sum(!data$is_header[seq_len(i)], na.rm = TRUE)
      if (data_seq_i %% 2 == 0) {
        grid.rect(
          x      = unit((margin_left + table_right) / 2, "npc"),
          y      = unit(y_mid, "npc"),
          width  = unit(table_right - margin_left, "npc"),
          height = unit(row_h, "npc"),
          gp     = gpar(fill = "#F4F4EF", col = NA)
        )
      }
      grid.text(
        as.character(data$Phenotype[i]),
        x    = unit(col_x_left[1] + 0.005, "npc"),
        y    = unit(y_mid, "npc"),
        just = c("left", "center"),
        gp   = gpar(fontsize = font_size, col = "black")
      )
      grid.text(
        as.character(data$ICD10[i]),
        x    = unit(col_x_center[2], "npc"),
        y    = unit(y_mid, "npc"),
        just = c("center", "center"),
        gp   = gpar(fontsize = font_size, col = "black")
      )
    }
    current_y <- y_bot
  }
  grid.lines(
    x  = unit(c(margin_left, table_right), "npc"),
    y  = unit(c(current_y, current_y), "npc"),
    gp = gpar(lwd = 1.2, col = "black")
  )
  dev.off()
}








