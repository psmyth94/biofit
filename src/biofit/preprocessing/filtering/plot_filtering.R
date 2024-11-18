source(file.path(R_SCRIPTS_PATH, "plotting_utils.R"))


plot_filter <- function(
    list_of_sums, path,
    xlabs,
    mains,
    legend_position = "top",
    legend_title = "Filtering",
    before_name = "Before",
    after_name = "After",
    xlog = NULL,
    ylog = NULL,
    ...) {
  
  
  suppressPackageStartupMessages(require(ggplot2))
  suppressPackageStartupMessages(require(RColorBrewer))
  suppressPackageStartupMessages(require(circlize))
  suppressPackageStartupMessages(require(patchwork))

  if (!is.list(list_of_sums) && !is.vector(list_of_sums)) {
    list_of_sums <- list(list_of_sums)
  }

  if (!is.list(xlabs) && !is.vector(xlabs)) {
    xlabs <- list(xlabs)
  }

  if (!is.list(mains) && !is.vector(mains)) {
    mains <- list(mains)
  }

  if (length(xlabs) != length(list_of_sums) && length(xlabs) == 1) {
    xlabs <- rep(xlabs, length(list_of_sums))
  }

  if (length(mains) != length(list_of_sums) && length(mains) == 1) {
    mains <- rep(mains, length(list_of_sums))
  }
  plots <- NULL
  for (i in seq_along(list_of_sums)) {
    x1 <- as.vector(list_of_sums[[i]][[1]])
    x2 <- as.vector(list_of_sums[[i]][[2]])
    if (is.null(plots)) {
      plots <- generate_comparison_histogram(
        x1, x2,
        xlab = xlabs[[i]], title = mains[[i]], xlog = xlog, ylog = ylog,
        legend_title = legend_title, subplot_title1 = before_name,
        subplot_title2 = after_name
      )
    } else {
      plots <- plots + generate_comparison_histogram(
        x1, x2,
        xlab = xlabs[[i]], title = mains[[i]], xlog = xlog, ylog = ylog,
        legend_title = legend_title, subplot_title1 = before_name,
        subplot_title2 = after_name
      )
    }
  }

  grid <- plots + patchwork::plot_layout(guides = "collect", ncol = ncol) &
    ggplot2::theme(
      text = ggplot2::element_text(size = 8),
      legend.position = legend_position,
    )

  save_plots(path, plot = grid, width = 6, height = 6.5, dpi = 600)
}
