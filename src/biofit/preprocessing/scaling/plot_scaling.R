source(file.path(R_SCRIPTS_PATH, "plotting_utils.R"))


plot_scaler <- function(
    x1,
    x2 = NULL,
    y1 = NULL,
    y2 = NULL,
    path = NULL,
    input_columns1 = NULL,
    input_columns2 = NULL,
    label_name1 = NULL,
    label_name2 = NULL,
    ylab = NULL,
    xlab = NULL,
    ylim = NULL,
    before_title = NULL,
    after_title = NULL,
    legend_position = NULL,
    add_box = TRUE,
    horizontal_plot = FALSE,
    order = FALSE,
    col_set = "Set1",
    cols = NULL,
    log_num = NULL,
    show_outliers = TRUE) {

  suppressPackageStartupMessages(require(patchwork))
  
  if (is.null(y2)) {
    y2 <- y1
  }
  p1 <- generate_violin(
    x1, y1,
    column = input_columns1, label_name = label_name1, ylim = ylim,
    xlab = xlab, ylab = ylab, title = before_title, legend_position = legend_position,
    add_box = add_box, horizontal_plot = horizontal_plot, order = order,
    col_set = col_set, cols = cols, log_num = log_num, show_outliers = show_outliers
  )
  if (!is.null(x2)) {
    p2 <- generate_violin(
      x2, y2,
      column = input_columns2, label_name = label_name2, ylim = ylim,
      xlab = xlab, ylab = ylab, title = after_title, legend_position = legend_position,
      add_box = add_box, horizontal_plot = horizontal_plot, order = order,
      col_set = col_set, cols = cols, log_num = log_num, show_outliers = show_outliers
    )
    the_plot <- p1 / p2
  } else {
    the_plot <- p1
  }
  if (!is.null(path)) {
    save_plots(path, plot = the_plot, width = 6, height = 6, dpi = 600)
  }
  return(path)
}
