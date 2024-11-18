source()

plot_rfe_feature_selector_for_genomics <- function(
    x1, x2,
    legend_title = "Feature Selection by Recursive Feature Elimination",
    ...) {
    args <- list(...)
    args$x1 <- x1
    args$x2 <- x2
    args$legend_title <- if ("legend_title" %in% names(args)) args$legend_title else legend_title
    do.call(plot_feature_selector_for_genomics, args)
}


plot_rfe_feature_selector_snp <- function(
    x1, x2,
    feature_main = "By GenomicVariantss",
    ...) {
    args <- list(...)
    args$x1 <- x1
    args$x2 <- x2
    args$feature_main <- if ("feature_main" %in% names(args)) args$feature_main else feature_main
    do.call(plot_rfe_feature_selector_for_genomics, args)
}

