# Full path: src/biofit/preprocessing/scaling/tmm/tmm.R
# adapted from ruppinlab/sklearn-extensions and edgeR::calcNormFactors source code

source(file.path(R_SCRIPTS_PATH, "utils.R"))

edger_tmm_ref_column <- function(counts, lib_size=colSums(counts), p=0.75) {
  y <- t(t(counts) / lib_size)
  f <- apply(y, 2, function(x) quantile(x, p=p))
  ref_column <- which.min(abs(f - mean(f)))
}

edger_tmm_fit <- function(X) {
  suppressPackageStartupMessages(require(edgeR))
  suppressPackageStartupMessages(require(arrow))
  X <- as.matrix(as.data.frame(X))
  counts <- t(X)
  ref_sample <- counts[, edger_tmm_ref_column(counts)]
  return(ref_sample)
}

edger_tmm_cpm_transform <- function(X, ref_samples, log=TRUE, prior_count=2) {
  suppressPackageStartupMessages(require(edgeR))
  suppressPackageStartupMessages(require(arrow))
  X <- as.matrix(convert_to_dataframe(X))
  counts <- t(X)
  ref_sample_mask <- apply(counts, 2, function(c) all(c == ref_samples))
  if (any(ref_sample_mask)) {
    dge <- edgeR::DGEList(counts=counts)
    dge <- edgeR::calcNormFactors(
      dge, method="TMM", refColumn=min(which(ref_sample_mask))
    )
    cpms <- edgeR::cpm(dge, log=log, prior.count=prior_count)
  } else {
    counts <- cbind(counts, ref_samples)
    colnames(counts) <- NULL
    dge <- edgeR::DGEList(counts=counts)
    dge <- edgeR::calcNormFactors(dge, method="TMM", refColumn=ncol(dge))
    cpms <- edgeR::cpm(dge, log=log, prior.count=prior_count)
    cpms <- cpms[, -ncol(cpms)]
  }
  return(arrow::as_arrow_table(as.data.frame(t(cpms))))
}

edger_tmm_tpm_transform <- function(
    X, feature_meta, ref_samples, log=TRUE, prior_count=2, meta_col="Length"
) {
  if (is.null(feature_meta)) stop("feature_meta cannot be NULL")
  suppressPackageStartupMessages(require(edgeR))
  suppressPackageStartupMessages(require(arrow))
  X <- as.matrix(convert_to_dataframe(X))
  counts <- t(X)
  ref_sample_mask <- apply(counts, 2, function(c) all(c == ref_samples))
  if (any(ref_sample_mask)) {
    dge <- edgeR::DGEList(counts=counts, genes=feature_meta)
    dge <- edgeR::calcNormFactors(
      dge, method="TMM", refColumn=min(which(ref_sample_mask))
    )
  } else {
    counts <- cbind(counts, ref_samples)
    colnames(counts) <- NULL
    dge <- edgeR::DGEList(counts=counts, genes=feature_meta)
    dge <- edgeR::calcNormFactors(dge, method="TMM", refColumn=ncol(dge))
  }
  if (log) {
    # XXX: edgeR doesn't have built-in support for logTPM w/ prior.count
    #      so do API internal logic manually
    # TODO: use effectiveLibSizes() in newer edgeR versions
    lib_size <- dge$samples$lib.size * dge$samples$norm.factors
    scaled_prior_count <- args$prior_count * lib_size / mean(lib_size)
    adj_lib_size <- lib_size + 2 * scaled_prior_count
    fpkms <- t(
      (t(dge$counts) + scaled_prior_count) / adj_lib_size
    ) * 1e6 / dge$genes[[meta_col]] * 1e3
    tpms <- log2(t(t(fpkms) / colSums(fpkms)) * 1e6)
  } else {
    fpkms <- edgeR::rpkm(
      dge, gene.length=meta_col, log=log, prior.count=prior_count
    )
    tpms <- t(t(fpkms) / colSums(fpkms)) * 1e6
  }
  if (!any(ref_sample_mask)) tpms <- tpms[, -ncol(tpms)]
  return(arrow::as_arrow_table(t(tpms)))
}

edger_cpm_transform <- function(X, log=TRUE, prior_count=2) {
  suppressPackageStartupMessages(require(edgeR))
  suppressPackageStartupMessages(require(arrow))
  return(arrow::as_arrow_table(t(edgeR::cpm(t(X), log=log, prior.count=prior_count))))
}
