# ============================================================
# Extended Data Table 2&3
# ============================================================ 

library(dplyr)
library(tidyr)
library(grid)
MODEL1_CSV  <- ".csv"
MODEL2_CSV  <- ".csv"
PHENO_MAP <- c()
NEONATAL_PHENOS <- c()
TARGET_COL_ORDER <- c()
read_and_pivot <- function(csv_path, model_label) {
  df <- read.csv(csv_path, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
  df <- df %>%
    mutate(
      Mean_num = suppressWarnings(as.numeric(as.character(Mean))),
      SD_num   = suppressWarnings(as.numeric(as.character(Standard.Deviation))),
      display  = mapply(
        FUN = function(m, s, m_raw, s_raw) {
          m_missing <- is.na(m) || is.na(m_raw) || trimws(m_raw) %in% c("", "-")
          s_missing <- is.na(s) || is.na(s_raw) || trimws(s_raw) %in% c("", "-")
          if (!m_missing && !s_missing) {
            paste0(sprintf("%.4f", round(m, 4)), " \u00b1 ", sprintf("%.4f", round(s, 4)))
          } else if (!m_missing && s_missing) {
            sprintf("%.4f", round(m, 4))
          } else {
            "-"
          }
        },
        Mean_num, SD_num, Mean, Standard.Deviation,
        USE.NAMES = FALSE
      )
    ) %>%
    select(-Mean_num, -SD_num)
  dup <- df %>%
    group_by(Phenotype, Metric) %>%
    filter(n() > 1)
  if (nrow(dup) > 0) {
    dup_summary <- dup %>%
      summarise(n = n(), .groups = "drop") %>%
      as.data.frame()
    df <- df %>% group_by(Phenotype, Metric) %>% slice_head(n = 1) %>% ungroup()
  }
  wide <- df %>%
    select(Phenotype, Metric, display) %>%
    tidyr::pivot_wider(
      names_from  = Metric,
      values_from = display,
      values_fn   = function(x) { as.character(x[1]) }
    )
  col_order <- TARGET_COL_ORDER
  missing_cols <- setdiff(col_order, names(wide))
  present_cols <- intersect(col_order, names(wide))
  wide <- wide[, present_cols, drop = FALSE]
  wide$is_neonatal <- wide$Phenotype %in% NEONATAL_PHENOS
  return(wide)
}
insert_section_headers <- function(wide) {
  rows_list <- vector("list", nrow(wide) + 2)
  has_maternal  <- any(!wide$is_neonatal)
  has_neonatal <- any(wide$is_neonatal)
  col_names <- names(wide)
  ri <- 1
  if (has_maternal && has_neonatal) {
    header_row <- setNames(rep(list(as.character(NA)), length(col_names)), col_names)
    header_row[[which(col_names == "Phenotype_en")]]  <- "Maternal Phenotypes"
    rows_list[[ri]] <- header_row; ri <- ri + 1
    for (i in seq_len(nrow(wide))) {
      if (!wide$is_neonatal[i]) {
        rows_list[[ri]] <- as.list(wide[i, ]); ri <- ri + 1
      }
    }
    header_row2 <- setNames(rep(list(as.character(NA)), length(col_names)), col_names)
    header_row2[[which(col_names == "Phenotype_en")]]   <- "Neonatal Phenotypes"
    rows_list[[ri]] <- header_row2; ri <- ri + 1
    for (i in seq_len(nrow(wide))) {
      if (wide$is_neonatal[i]) {
        rows_list[[ri]] <- as.list(wide[i, ]); ri <- ri + 1
      }
    }
  } else {
    for (i in seq_len(nrow(wide))) {
      rows_list[[i]] <- as.list(wide[i, ])
    }
    ri <- nrow(wide) + 1
  }
  out <- do.call(rbind, lapply(rows_list[seq_len(ri - 1)], function(r) {
    data.frame(r, stringsAsFactors = FALSE)
  }))
  return(out)
}
draw_performance_table <- function(
    data,
    title_text,
    out_pdf,
    font_size    = 7.5,
    header_size  = 8,
    title_size   = 10,
    page_width   = 8.27,      
    page_height  = 11.69       
) {
  nr  <- nrow(data)
  nc  <- ncol(data)
  raw_names <- names(data)
  col_labels <- raw_names
  display_col_idx <- which(!raw_names %in% c("Phenotype_en", "is_neonatal"))
  nd <- length(display_col_idx)
  margin_left  <- 0.10                     
  margin_right <- 0.10                     
  usable_w     <- 1 - margin_left - margin_right  
  pheno_rel    <- 0.28
  metric_rel   <- (1 - pheno_rel) / (nd - 1)
  col_widths   <- c(pheno_rel * usable_w,
                    rep(metric_rel * usable_w, nd - 1))
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
  n_section   <- sum(data$Phenotype_en %in%
                       c("Maternal Phenotypes", "Neonatal Phenotypes"), na.rm = TRUE)
  n_data      <- nr - n_section
  total_needed <- header_h + n_section * section_h + n_data * normal_row_h + 0.06
  available   <- table_top - 0.04
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
  for (j in seq_len(nd)) {
    lbl <- col_labels[display_col_idx[j]]
    grid.text(
      lbl,
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
    is_section <- !is.na(data$Phenotype_en[i]) &&
      data$Phenotype_en[i] %in% c("Maternal Phenotypes", "Neonatal Phenotypes")
    row_h <- if (is_section) section_h else normal_row_h
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
        data$Phenotype_en[i],
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
      data_seq_i <- sum(
        !data$Phenotype_en[seq_len(i)] %in%
          c("Maternal Phenotypes", "Neonatal Phenotypes"),
        na.rm = TRUE
      )
      if (data_seq_i %% 2 == 0) {
        grid.rect(
          x      = unit((margin_left + table_right) / 2, "npc"),
          y      = unit(y_mid, "npc"),
          width  = unit(table_right - margin_left, "npc"),
          height = unit(row_h, "npc"),
          gp     = gpar(fill = "#F4F4EF", col = NA)
        )
      }
      for (j in seq_len(nd)) {
        col_name <- raw_names[display_col_idx[j]]
        cell_val <- as.character(data[i, col_name])
        justify <- if (j == 1) c("left", "center") else c("center", "center")
        x_pos   <- if (j == 1) col_x_left[j] + 0.005 else col_x_center[j]
        grid.text(
          display_text,
          x    = unit(x_pos, "npc"),
          y    = unit(y_mid, "npc"),
          just = justify,
          gp   = gpar(fontsize = font_size, col = "black")
        )
      }
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

