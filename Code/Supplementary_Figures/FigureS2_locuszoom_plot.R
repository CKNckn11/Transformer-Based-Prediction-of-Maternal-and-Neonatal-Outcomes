# ============================================================
# locuszoom plot
# ============================================================ 
library(locuszoomr)
library(data.table)
library(EnsDb.Hsapiens.v86)
tryCatch({
  f <- get("scatter_plot", envir = asNamespace("locuszoomr"))
  src <- deparse(f, width.cutoff = 500)
  src <- gsub(
    "mar = c(ifelse(xticks, 3, 0.1), 3.5,",
    "mar = c(ifelse(xticks, 3, 0.1), 4.0,",
    src, fixed = TRUE
  )
  src <- gsub(
    "mgp = c(1.7, 0.5, 0)",
    "mgp = c(2.0, 0.5, 0)",
    src, fixed = TRUE
  )
  f_new <- eval(parse(text = paste(src, collapse = "\n")))
  environment(f_new) <- asNamespace("locuszoomr")
  assignInNamespace("scatter_plot", f_new, ns = asNamespace("locuszoomr"))
  cat("鈽卹[2]=3.5鈫?.5, mgp[1]=1.7鈫?.5\n")
}, error = function(e) {
  cat(conditionMessage(e), "\n")
})
gwas_dir <- "/400T/ShuChang/ilus/GEMMA/TestLMM0/LMM4_re"
clump_dir <- file.path(
  gwas_dir,
  "clumping_results"
)
ld_dir <- file.path(
  gwas_dir,
  "ld_results"
)
out_dir <- file.path(
  gwas_dir,
  "locuszoom_plots_final6.20"
)
dir.create(
  out_dir,
  recursive = TRUE,
  showWarnings = FALSE
)
FLANK_BP <- 5e5
TOP_N_SNPS <- 10
PDF_WIDTH <- 13
PDF_HEIGHT <- 5
GENE_TRACKS <- 3
partner_colors <- c(
  "#14128c",
  "#29d8ca",
  "#065f1a",
  "#ec7807",
  "#fc1403"
)
collapse_gene_isoforms <- function(loc, lead_pos = NULL, max_genes = 10) {
  tx <- loc$TX
  if (is.null(tx) || nrow(tx) == 0) return(loc)
  gene_name_char <- as.character(tx$gene_name)
  na_idx <- is.na(gene_name_char) | gene_name_char == "NA" | gene_name_char == ""
  if (any(na_idx)) {
    gene_name_char[na_idx] <- paste0("__na_", seq_len(sum(na_idx)), "__")
  }
  tx$gene_name_char <- gene_name_char
  tx <- tx[order(tx$gene_name_char, -tx$width), ]
  tx <- tx[!duplicated(tx$gene_name_char), ]
  tx$gene_name_char <- NULL  
  biotype <- tolower(as.character(tx$gene_biotype))
  l1_types <- c("protein_coding")
  l2_types <- c("protein_coding", "lncrna", "mirna", "snorna", "snrna")
  l3_types <- c(l2_types, "antisense", "sense_intronic", "misc_rna", "processed_transcript")
  l1 <- tx[biotype %in% l1_types, ]
  l2 <- tx[biotype %in% l2_types, ]
  l3 <- tx[biotype %in% l3_types, ]
  if (nrow(l1) > 0) {
    tx_f <- l1
    flvl <- "protein_coding"
  } else if (nrow(l2) > 0) {
    tx_f <- l2
    flvl <- "protein_coding+lncRNA/miRNA"
  } else if (nrow(l3) > 0) {
    tx_f <- l3
    flvl <- "protein_coding+lncRNA/miRNA+antisense"
  } else {
    tx_f <- tx
    flvl <- "all (no filter)"
  }
  gene_names <- as.character(tx_f$gene_name)
  bac_pattern <- grepl("^(RP|AC|AL|AP|BX|CTD|KB)[0-9]", gene_names, ignore.case = TRUE)
  non_bac <- tx_f[!bac_pattern, ]
  if (nrow(non_bac) > 0) {
    tx_f <- non_bac
    flvl <- paste0(flvl, " (non-BAC)")
  } else {
    flvl <- paste0(flvl, " (BAC-only fallback)")
  }
  if (nrow(tx_f) > max_genes && !is.null(lead_pos)) {
    tx_mid <- (tx_f$start + tx_f$end) / 2
    dist_to_lead <- abs(tx_mid - lead_pos)
    tx_f <- tx_f[order(dist_to_lead), ]
    tx_f <- tx_f[seq_len(min(max_genes, nrow(tx_f))), ]
    cat(sprintf("  [gene filter] %s -> top %d by distance\n", flvl, max_genes))
  } else {
    cat(sprintf("  [gene filter] %s -> %d genes\n", flvl, nrow(tx_f)))
  }
  loc$TX <- tx_f
  return(loc)
}
clumped_files <- list.files(
  clump_dir,
  pattern = "\\.clumped$",
  full.names = TRUE
)
for (clump_file in clumped_files) {
  pheno <- gsub(
    "_clumped\\.clumped$",
    "",
    basename(clump_file)
  )
  clump <- tryCatch(
    fread(clump_file),
    error = function(e) NULL
  )
  if (is.null(clump) || nrow(clump) == 0) {
    cat("?? ")
    next
  }
  clump <- clump[order(P)]
  n_snps <- min(TOP_N_SNPS, nrow(clump))
  lead_snps <- clump[
    1:n_snps,
    .(SNP, CHR, BP)
  ]
  gwas_file <- file.path(
    gwas_dir,
    paste0(pheno, "_gwas.assoc.txt")
  )
  if (!file.exists(gwas_file)) {
    cat("?? :\n")
    cat(gwas_file, "\n")
    next
  }
  gwas <- tryCatch({
    fread(gwas_file)
  }, error = function(e) {
    NULL
  })
  if (is.null(gwas)) {
    cat("??")
    next
  }
  setnames(
    gwas,
    c("chr", "ps", "p_lrt", "rs"),
    c("CHR", "BP", "P", "SNP"),
    skip_absent = TRUE
  )
  gwas[, CHR := as.integer(CHR)]
  gwas[, BP := as.integer(BP)]
  gwas <- gwas[!is.na(P)]
  for (i in 1:nrow(lead_snps)) {
    snp <- lead_snps$SNP[i]
    chr <- as.integer(lead_snps$CHR[i])
    bp  <- as.integer(lead_snps$BP[i])
    cat("\n--------------------------------------\n")
    cat("Lead SNP:", snp, "\n")
    cat("--------------------------------------\n")
    ld_file <- file.path(
      ld_dir,
      paste0(snp, "_ld.ld")
    )
    if (!file.exists(ld_file)) {
      cat("?? ")
      next
    }
    ld <- tryCatch({
      fread(ld_file)
    }, error = function(e) {
      NULL
    })
    if (is.null(ld) || nrow(ld) == 0) {
      cat("?? ")
      next
    }
    setnames(
      ld,
      c("SNP_B", "R2"),
      c("SNP", "r2"),
      skip_absent = TRUE
    )
    ld <- ld[CHR_B == chr]
    region_gwas <- gwas[
      CHR == chr &
        BP >= (bp - FLANK_BP) &
        BP <= (bp + FLANK_BP)
    ]
    if (nrow(region_gwas) < 5) {
      cat("??")
      next
    }
    region <- merge(
      region_gwas,
      ld[, .(SNP, r2)],
      by = "SNP",
      all.x = TRUE
    )
    region[is.na(r2), r2 := 0]
    region[SNP == snp, r2 := 1]
    if (nrow(region) < 5) {
      cat("??")
      next
    }
    cat(
      sprintf(
        nrow(region),
        max(-log10(region$P), na.rm = TRUE)
      )
    )
    tryCatch({
      loc <- locus(
        data = region,
        ens_db = EnsDb.Hsapiens.v86,
        chrom = "CHR",
        pos   = "BP",
        p     = "P",
        labs = "SNP",
        index_snp = snp,
        flank = FLANK_BP,
        LD = "r2"
      )
      lead_pos <- region[SNP == snp, BP][[1]]
      loc <- collapse_gene_isoforms(loc, lead_pos = lead_pos, max_genes = 10)
      n_genes <- if (!is.null(loc$TX)) nrow(loc$TX) else 0
      cat(sprintf(": %d\n", n_genes))
      outfile <- file.path(
        out_dir,
        paste0(
          pheno,
          "_",
          snp,
          "_locuszoom.pdf"
        )
      )
      cairo_pdf(
        outfile,
        width = PDF_WIDTH,
        height = PDF_HEIGHT,
        family = "Arial"
      )
      locus_plot(
        loc,
        showLD = TRUE,
        legend_pos = NULL,
        main = paste(snp),
        ld_colors = partner_colors,
        point.size = 5.8,
        highlight_index = TRUE,
        index_snp_color = "purple",
        index_snp_shape = 23,
        index_snp_size  = 6.5,
        ylim = c(
          0,
          max(-log10(region$P), na.rm = TRUE) * 1.15
        ),
        cex = 1.8,
        cex.axis = 1.6,
        cex.lab = 1.8,
        cex.text = 1.5,
        xlab = paste0("Chromosome ", chr),
        italics = TRUE,
        ylab = expression(-log[10]~(italic(P))),
        maxrows = GENE_TRACKS,
        genetracks = GENE_TRACKS
      )
      dev.off()
      cat("? done:\n")
      cat(outfile, "\n")
    }, error = function(e) {
      cat("? locuszoom:\n")
      cat(e$message, "\n")
    })
  }
}




