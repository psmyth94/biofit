source(file.path(R_SCRIPTS_PATH, "utils.R"))

sci_format <- function(x) {
  suppressPackageStartupMessages(require(scales))
  ifelse((abs(x) < 1 & abs(x) > 0 & floor(log10(abs(x))) <= -2) | abs(x) >= 10000,
    formatC(x, format = "e", digits = 2), formatC(x, format = "f", digits = 2)
  )
}

#' `prepare_data_for_hist()` calculates row sums and columns sums for 1 or 2 data frames.
#'
#' Note: If you leave x2 blank, it will be NULL and not calculate.
#'
#' @param x1 data.frame: 1st or only data frame to have sums be calculated
#' @param x2 data.frame: 2nd data frame to have sums be calculated. Default is NULL.
#'
#' @returns list: a list of all the sums; 1st two values in the list are for x1, the next two for x2.
#'
prepare_data_for_hist <- function(x1, x2 = NULL) {
  # Calculate row sum and col sums of dataset 1
  data1_sample <- rowSums(x1)
  data1_feature <- colSums(x1)

  list_sums <- list(data1_sample, data1_feature)

  # Calculates row and col sums for second dataset if it is given Allows for
  # situations with only a single dataset
  if (!is.null(x2)) {
    data2_sample <- rowSums(x2)
    data2_feature <- colSums(x2)

    list_sums <- append(list_sums, list(data2_sample, data2_feature))
  }

  return(list_sums)
}

#' `non_zero_sums()` calculates row sums and columns sums for non-zero values in 1 or 2 data frames.
#'
#' Note: If you leave x2 blank, it will be NULL and not calculate.
#'
#' @param x1 (data.frame) 1st or only data frame to have non-zero sums be calculated.
#' @param x2 (data.frame) 2nd data frame to have non-zero sums be calculated. Default is NULL.
#'
#' @returns (list) of all the sums; 1st value in the list are for x1, the 2nd is for x2 if used.
#'
non_zero_sums <- function(x1, x2 = NULL) {
  # Sum all non-zero values for a dataset
  x1_non_zero_row <- rowSums(x1 != 0)
  x1_non_zero_col <- colSums(x1 != 0)
  sums <- list(x1_non_zero_row, x1_non_zero_col)

  # Allows for situations with only a single dataset
  if (!is.null(x2)) {
    x2_non_zero_row <- rowSums(x2 != 0)
    x2_non_zero_col <- colSums(x2 != 0)
    sums <- append(sums, list(x2_non_zero_row, x2_non_zero_col))
  }

  # Returns list of non_zero rowSums
  return(sums)
}

#' `prepare_axis_label()` adjusts axis label names to include log transformation.
#'
#' @param label chr: the axis label to be adjusted.
#' @param log_type chr: the log transformation being applied.
#'
#' @returns chr: the new label with the transformation added.
#'
prepare_axis_label <- function(label, log_type) {
  if (grepl("1p", log_type)) {
    if (grepl("_1p", log_type)) {
      label_log <- gsub("_1p", "", log_type)
      label <- paste0(label, " (", label_log, "(x+1))")
    } else {
      label <- paste0(label, " (ln(x+1))")
    }
  } else if (log_type == "log") {
    label <- paste0(label, " (ln)")
  } else {
    label <- paste0(label, " (", log_type, ")")
  }
  return(label)
}

log_transformation <- function(x, log_type) {
  if (grepl("1p", log_type)) {
    if (grepl("_1p", log_type)) {
      label_log <- gsub("_1p", "", log_type)
      if (label_log == "log10") {
        return(log10(1 + x))
      } else if (label_log == "log2") {
        return(log2(1 + x))
      }
    } else {
      return(log(1 + x))
    }
  } else if (log_type == "log") {
    return(log(x))
  } else if (log_type == "log2") {
    return(log2(x))
  } else if (log_type == "log10") {
    return(log10(x))
  }
  return(x)
}

#'
#' Transformations functions for different transformation functions
#' They are called by scale_*_continuous when we enter the name (first argument in trans_new())
#' They just need to be here, don't need to be called any other time.
#'

#' `log1p()` is a transformation function for log with x + 1 as an input
log2_1p <- function(x) {
  log2(1 + x)
}

#' `log2_1p_trans()` is a transformation function for log base 2 with x + 1 as an input
log2_1p_trans <- function() {
  suppressPackageStartupMessages(require(scales))

  trans_new("log2_1p",
    transform = log2_1p, inverse = function(x) {
      2^x - 1
    }, breaks = trans_breaks(log2_1p, function(x) 2^x - 1),
    domain = c(0, Inf)
  )
}

#' `log10_1p()` is a transformation function for log base 10 with x + 1 as an input
log10_1p <- function(x) {
  log10(1 + x)
}

#' `log10_1p_trans()` is a transformation function for log base 10 with x + 1 as an input
log10_1p_trans <- function() {
  suppressPackageStartupMessages(require(scales))

  trans_new("log10_1p",
    transform = log10_1p, inverse = function(x) {
      10^x - 1
    }, breaks = trans_breaks(log10_1p, function(x) 10^x - 1),
    domain = c(0, Inf)
  )
}

#'
#' `color_select()` prepares a vector of colours to use in a ggplot
#'
#' Note: Requires RColorBrewer and circulize packages
#'
#' @details It will use RColorBrewer's sets of colours as the base for the vector
#' If the number of colours needed are <= the number of colors in the set,
#' then the colours are used directly from the set
#' Else, the function will generate colors in between using a colour ramp
#'
#' @param levels (int) number of colours that are to be returned
#' @param col_set (chr) the RColorBrewer set that is to be used, and only those sets
#'  ('Set1', 'Set2', 'Set3', 'Pastel2', 'Pastel1', 'Paired', 'Dark2', 'Accent')
#'
#' @returns (vec) a vector of colours
#'
#' @examples
#' \dontrun{
#' colors <- color_select(5, "Set3")
#' }
color_select <- function(levels, col_set = "Set1") {
  col_set <- match.arg(col_set, c(
    "Set1", "Set2", "Set3", "Pastel2", "Pastel1",
    "Paired", "Dark2", "Accent"
  ))

  if (col_set %in% c("Set1", "Pastel1")) {
    num_col <- 9
  } else if (col_set %in% c("Set3", "Paired")) {
    num_col <- 12
  } else {
    num_col <- 8
  }

  if (levels > num_col) {
    at <- seq(0, levels, length.out = num_col)
    color_fn <- circlize::colorRamp2(at, RColorBrewer::brewer.pal(num_col, col_set))
    colors <- color_fn(1:levels)
  } else {
    color_fn <- RColorBrewer::brewer.pal(num_col, col_set)
    colors <- color_fn[1:levels]
  }
  return(colors)
}


#' Generate a simple histogram for one variable.
#'
#' `generate_histogram()` creates a simple histogram for a single numerical variable.
#'
#' Note: requires packages ggplot2 and rlang.
#'
#' @param data (data.frame or vector) a data frame that the function is obtaining the data from or a vector of data
#'  If you pass a vector, it will be made into a data.frame with the label as the name of the column.
#' @param column (chr) the column with the data that is to be used in the histogram.
#'  Default is value so label doesn't need to be provided and the function works.
#' @param xlab (chr) name of the x-axis. Default is 'X'.
#' @param ylab (chr) name of the y-axis. Default is 'Frequency'.
#' @param title (chr) title of the figure. Default is 'Histogram'.
#' @param bins (int) number of bins for the histogram. Default is 30.
#' @param font_size (dbl) size of the font for the plot. Default is 8.
#' @param alpha (dbl) Opacity of bars. Default is 0.6.
#' @param col_fill (chr) primary colour of the bars. Default is 'grey40'.
#' @param col_outline (chr) colour of the outline of the bars. Default is 'black'.
#' @param xlog (chr) logarithmic transformation of the x-axis.
#'   Options include: (default) NULL (none), 'log', 'log2', 'log10', 'log1p', 'log2_1p', 'log10_1p'
#' @param ylog (chr) logarithmic transformation of the y-axis. Default is NULL (none).
#'   Options include: (default) NULL (none), 'log', 'log2', 'log10', 'log1p', 'log2_1p', 'log10_1p'
#'
#' @returns the ggplot build of the final histogram.
#'
#' @examples
#' \dontrun{
#' generate_histogram(mtcars, wt)
#'
#' generate_histogram(mtcars, wt,
#'   xlab = "Weights", ylab = "Frequency",
#'   title = "Weights of Cars", bins = 30,
#'   col_fill = "red", col_outline = "black", xlog = "log2",
#'   ylog = NULL
#' )
#' }
generate_histogram <- function(
    data, column = NULL, xlab = "X", ylab = "Frequency",
    title = "Histogram", bins = 30, font_size = 8, alpha = 0.6,
    col_fill = "grey40", col_outline = "black",
    xlog = NULL, ylog = NULL) {
  data <- convert_to_dataframe(data)

  column <- get_default_columns(data, column)
  if (is.null(column)) {
    data <- reshape2::melt(data)
    column <- "value"
  }
  data <- validate_data(data, column)

  if (!is.null(xlog)) {
    xlog <- match.arg(xlog, c(
      "log2", "log10", "log", "log2_1p", "log10_1p",
      "log1p"
    ))
    xlab <- prepare_axis_label(xlab, xlog)
  }
  if (!is.null(ylog)) {
    ylog <- match.arg(ylog, c(
      "log2", "log10", "log", "log2_1p", "log10_1p",
      "log1p"
    ))
    ylab <- prepare_axis_label(ylab, ylog)
  }

  suppressPackageStartupMessages(require(ggplot2))

  ggplot(data, aes(x = .data[[column]])) +
    geom_histogram(
      bins = bins, fill = col_fill,
      color = col_outline, alpha = alpha
    ) +
    labs(x = xlab, y = ylab, title = title) +
    theme_bw() +
    theme(text = element_text(size = font_size), plot.title = element_text(hjust = 0.5)) +
    {
      if (!is.null(xlog)) {
        scale_x_continuous(trans = xlog)
      }
    } +
    {
      if (!is.null(ylog)) {
        scale_y_continuous(trans = ylog)
      }
    }
}

#' Generate a histogram to compare two sets of values
#'
#' `generate_comparison_histogram()` creates a histogram that compares values from two groups. Typically a before and after some transformation or filtering, but could also just be two categories.
#'
#' Note: requires packages ggplot2, RColorBrewer.
#'
#' @param data1 (vector) data of the 1st group.
#' @param data2 (vector) data of the 2nd group.
#' @param column1 (chr) name of column containing first set of data
#' @param column2 (chr) name of column containing second set of data
#' @param xlab (chr) name of the x-axis.
#' @param ylab (chr) name of the y-axis. Default is 'Count'.
#' @param title (chr) title of the figure. Default is 'Comparison Histogram'
#' @param bins (int) number of bins for the histogram. Default is 30.
#' @param alpha (double) opacity of the histograms. Value between 0 and 1. Default is 0.6.
#' @param legend_title (chr) name of the legend. Default is 'Feature Selection'.
#' @param legend_position (chr) placement of the legend in the figure.
#'  Options indude: (default) 'top', 'bottom', 'left', 'right'.
#' @param subplot_title1 (chr) name of the values from the 1st dataset. Default is 'Before'.
#' @param subplot_title2 (chr) name of the values from the 2nd dataset. Default is 'After'.
#' @param col_set (chr) name of RColorBrewer set to be used for colouring.
#'  Options include: (default) 'Set1', 'Set2', 'Set3', 'Pastel2', 'Pastel1', 'Paired', 'Dark2', 'Accent'
#' @param cols (chr (vec)) vector of colors that you wish to use. Default to NULL and other colours are produced using the set in col_set.
#' @param col_outline (chr) colour of the border for the bars. Default is 'black'.
#' @param xlog (chr) logarithmic transformation of the x-axis.
#'   Options include: (default) NULL (none), 'log', 'log2', 'log10'.
#' @param ylog (chr) logarithmic transformation of the y-axis. Default is NULL (none).
#'   Options include: (default) NULL (none), 'log', 'log2', 'log10'.
#'
#' @returns the ggplot build of the final histogram.
#'
#' @examples
#' \dontrun{
#' control <- PlantGrowth$weight[PlantGrowth$group == "ctrl"]
#' treat1 <- PlantGrowth$weight[PlantGrowth$group == "trt1"]
#' generate_comparison_histogram(control, treat1,
#'   xlab = "Weights",
#'   title = "Control VS. Treatment: Plant Weights"
#' )
#' }
generate_comparison_histogram <- function(
    data1, data2, column1 = NULL, column2 = NULL,
    xlab = NULL, ylab = "Count", title = "Comparison Histogram", bins = 30, alpha = 0.6,
    legend_title = "Feature Selection", legend_position = "top", subplot_title1 = "Before",
    subplot_title2 = "After", col_set = "Set1", cols = NULL, col_outline = "black", xlog = NULL,
    ylog = NULL, ...) {
  suppressPackageStartupMessages(require(ggplot2))

  data1 <- convert_to_dataframe(data1)

  data2 <- convert_to_dataframe(data2)

  column1 <- get_default_columns(data1, column1)
  if (is.null(column1)) {
    data1 <- reshape2::melt(data1)
    column1 <- "value"
  }
  data1 <- validate_data(data1, column1)

  column2 <- get_default_columns(data2, column2)
  if (is.null(column2)) {
    data2 <- reshape2::melt(data2)
    column2 <- "value"
  }
  data2 <- validate_data(data2, column1)

  if (is.null(xlab)) {
    if (is.null(column1)) {
      xlab <- "Values"
    } else {
      xlab <- column1
    }
  }

  if (!is.null(xlog)) {
    xlog <- match.arg(xlog, c(
      "log2", "log10", "log", "log2_1p", "log10_1p",
      "log1p"
    ))
    xlab <- prepare_axis_label(xlab, xlog)
  }
  if (!is.null(ylog)) {
    ylog <- match.arg(ylog, c(
      "log2", "log10", "log", "log2_1p", "log10_1p",
      "log1p"
    ))
    ylab <- prepare_axis_label(ylab, ylog)
  }

  data_frame <- rbind(data.frame(value = data1, category = subplot_title1), data.frame(
    value = data2,
    category = subplot_title2
  ))
  data_frame$category <- factor(data_frame$category, levels = c(subplot_title1, subplot_title2))

  # Default Colours
  if (is.null(cols)) {
    cols <- color_select(2, col_set = col_set)
    names(cols) <- c(subplot_title1, subplot_title2)
  }

  ggplot(data_frame, aes(x = .data[[column1]], fill = category)) +
    geom_histogram(
      bins = bins,
      alpha = alpha, position = "identity", color = col_outline
    ) +
    scale_fill_manual(values = c(
      cols[1],
      cols[2]
    ), name = legend_title) +
    labs(x = xlab, title = title) +
    theme_bw() +
    theme(
      legend.position = legend_position, text = element_text(size = 8),
      axis.title = element_text(size = 6), legend.text = element_text(size = 6),
      legend.key.size = unit(1, "line"), legend.title = element_text(size = 7)
    ) +
    {
      if (!is.null(xlog)) {
        scale_x_continuous(trans = xlog)
      }
    } +
    {
      if (!is.null(ylog)) {
        scale_y_continuous(trans = ylog)
      }
    }
}

#' Generate a simple density plot for one variable.
#'
#' `generate_density()` creates a simple density plot for a single numerical variable.
#'
#' Note: requires packages ggplot2.
#'
#' @param data (data.frame) a data frame that the function is obtaining the data from.
#' @param column (chr) the column with the data that is to be used in the density plot.
#' @param xlab (chr) name of the x-axis. Default is 'X'.
#' @param ylab (chr) name of the y-axis. Default is 'Density'.
#' @param title (chr) title of the figure. Default is 'Density Plot'.
#' @param col_fill (chr) primary colour of the bars. Default is 'grey40'.
#' @param col_outline (chr) colour of the outline of the bars. Default is 'black'.
#' @param adjust (dbl) adjust bandwidth, which determines how precise (lower values) or smooth (higher values) the density plot is. Default is 1.
#' @param alpha (dbl) opacity of the density plot. Default is 0.8.
#' @param xlog (chr) logarithmic transformation of the x-axis.
#'   Options include: (default) NULL (none), 'log', 'log2', 'log10', 'log1p', 'log2_1p', 'log10_1p'
#'
#' @returns the ggplot build of the final density plot.
#'
#' @examples
#' \dontrun{
#' generate_density(mtcars, wt)
#'
#' generate_density(mtcars, wt,
#'   xlab = "Weights", ylab = "Density",
#'   title = "Weights of Cars",
#'   col_fill = "red", col_outline = "black",
#'   adjust = 0.5, alpha = 0.8, xlog = "log2"
#' )
#' }
generate_density <- function(
    data, column = NULL, xlab = "X", ylab = "Density",
    title = "Density Plot", col_fill = "grey40", col_outline = "black", adjust = 1,
    alpha = 0.6, xlog = NULL) {
  suppressPackageStartupMessages(require(ggplot2))

  data <- convert_to_dataframe(data)

  column <- get_default_columns(data, column)
  if (is.null(column)) {
    data <- reshape2::melt(data)
    column <- "value"
  }
  data <- validate_data(data, column)

  if (!is.null(xlog)) {
    xlog <- match.arg(xlog, c(
      "log2", "log10", "log", "log2_1p", "log10_1p",
      "log1p"
    ))
    xlab <- prepare_axis_label(xlab, xlog)
  }

  if (!is.data.frame(data)) {
    data <- convert_to_dataframe(data)
    colnames(data) <- column
  }

  ggplot(data, aes(x = .data[[column]])) +
    geom_density(
      fill = col_fill, color = col_outline,
      adjust = adjust, alpha = alpha
    ) +
    labs(x = xlab, y = ylab, title = title) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5)) +
    {
      if (!is.null(xlog)) {
        scale_x_continuous(trans = xlog)
      }
    }
}

#' Generate a density plot to compare two sets of values.
#'
#' `generate_comparison_density()` creates a density plot that compares values from two groups. Typically a before and after some transformation or filtering, but could also just be two categories.
#'
#' Note: requires packages ggplot2, RColorBrewer.
#'
#' @param data1 (vector) data of the 1st group.
#' @param data2 (vector) data of the 2nd group.
#' @param column1 (chr) name of column containing first set of data
#' @param column2 (chr) name of column containing second set of data
#' @param xlab (chr) name of the x-axis.
#' @param ylab (chr) name of the y-axis. Default is 'Count'.
#' @param title (chr) title of the figure. Default is 'Comparison Density Plot'
#' @param legend_title (chr) name of the legend. Default is 'Feature Selection'.
#' @param legend_position (chr) placement of the legend in the figure.
#'  Options include: (default) 'top', 'bottom', 'left', 'right'.
#' @param subplot_title1 (chr) name of the values from the 1st dataset. Default is 'Before'.
#' @param subplot_title2 (chr) name of the values from the 2nd dataset. Default is 'After'.
#' @param col_set (chr) name of RColorBrewer set to be used for colouring.
#'  Options include: (default) 'Set1', 'Set2', 'Set3', 'Pastel2', 'Pastel1', 'Paired', 'Dark2', 'Accent'
#' @param cols (chr (vec)) vector of colors that you wish to use. Default to NULL and other colours are produced using the set in col_set.
#' @param col_outline (chr) colour of the border of the density shape. Default is 'black'.
#' @param adjust (dbl) adjust bandwidth, which determines how precise (lower values) or smooth (higher values) the density plot is. Default is 1.
#' @param alpha (dbl) opacity of the histograms. Value between 0 and 1. Default is 0.6.
#' @param xlog (chr) logarithmic transformation of the x-axis.
#'   Options include: (default) NULL (none), 'log', 'log2', 'log10', 'log1p', 'log2_1p', 'log10_1p'.
#'
#' @returns the ggplot build of the final density plot.
#'
#' @examples
#' \dontrun{
#' control <- PlantGrowth$weight[PlantGrowth$group == "ctrl"]
#' treat1 <- PlantGrowth$weight[PlantGrowth$group == "trt1"]
#' generate_comparison_density(control, treat1,
#'   xlab = "Weights",
#'   title = "Control VS. Treatment: Plant Weights"
#' )
#' }
generate_comparison_density <- function(
    data1, data2, column1 = NULL, column2 = NULL,
    xlab = NULL, ylab = "Count", title = "Comparison Density Plot", legend_title = "Feature Selection",
    legend_position = "top", subplot_title1 = "Before", subplot_title2 = "After", col_set = "Set1",
    cols = NULL, col_outline = "black", adjust = 1, alpha = 0.6, xlog = NULL) {
  suppressPackageStartupMessages(require(ggplot2))
  data1 <- convert_to_dataframe(data1)

  data2 <- convert_to_dataframe(data2)

  column1 <- get_default_columns(data1, column1)
  if (is.null(column1)) {
    data1 <- reshape2::melt(data1)
    column1 <- "value"
  }
  data1 <- validate_data(data1, column1)

  column2 <- get_default_columns(data2, column2)
  if (is.null(column2)) {
    data2 <- reshape2::melt(data2)
    column2 <- "value"
  }
  data2 <- validate_data(data2, column1)

  if (is.null(xlab)) {
    if (is.null(column1)) {
      xlab <- "Values"
    } else {
      xlab <- column1
    }
  }

  if (!is.null(xlog)) {
    xlog <- match.arg(xlog, c(
      "log2", "log10", "log", "log2_1p", "log10_1p",
      "log1p"
    ))
    xlab <- prepare_axis_label(xlab, xlog)
  }

  data_frame <- rbind(data.frame(value = data1, category = subplot_title1), data.frame(
    value = data2,
    category = subplot_title2
  ))
  data_frame$category <- factor(data_frame$category, levels = c(subplot_title1, subplot_title2))

  if (is.null(cols)) {
    cols <- color_select(2, col_set = col_set)
    names(cols) <- c(subplot_title1, subplot_title2)
  }

  ggplot(data_frame, aes(x = value, fill = category)) +
    geom_density(
      adjust = adjust,
      alpha = alpha, color = col_outline
    ) +
    scale_fill_manual(values = c(
      cols[1],
      cols[2]
    ), name = legend_title) +
    labs(x = xlab, title = title) +
    theme_bw() +
    theme(
      legend.position = legend_position, text = element_text(size = 8),
      axis.title = element_text(size = 6), legend.text = element_text(size = 6),
      legend.key.size = unit(1, "line"), legend.title = element_text(size = 7)
    ) +
    {
      if (!is.null(xlog)) {
        scale_x_continuous(trans = xlog)
      }
    }
}

#'
#' `generate_barplot()` creates a bar plot using ggplot2
#'
#' Note: Requires ggplot2 package
#'
#' @details Settings available for both normal or stacked (& proportional stacked) bar plots.
#' adding a groupby variable will make a stacked bar plot and setting prop = T will make proportional bar plot.
#'
#' @param data (data.frame/vector) data frame contained the data of interest or vector with levels (categorical variable)
#' @param y (data.frame or vector) contains the identity values/heights of the bars (ex. counts of the bars pre-calculated)
#' @param group
#' @param label_name (chr) the name of the categorical variable to be used for the bars. Default is 'Labels'.
#' @param value_name (chr) the name of the column containing values for y (pre-calculated counts)
#' @param groupby (chr) the name of the secondary categorical variable to group the bars. Default is NULL.
#' @param xlab (chr) x-axis label. Default is 'X'.
#' @param ylab (chr) y-axis label. Default is 'Count'
#' @param title (chr) plot name/title. Default is 'Bar Plot'
#' @param col_set (chr) name of RColorBrewer set to be used for colouring.
#'  Options include: (default) 'Set1', 'Set2', 'Set3', 'Pastel2', 'Pastel1', 'Paired', 'Dark2', 'Accent'
#' @param cols (chr (vec)) vector of colors that you wish to use. Default to NULL and other colours are produced using the set in col_set.
#' @param col_outline (chr) colour of the outline of the bars. Default is 'grey'.
#' @param col_labels (chr) colour of the labels on the bars. Default is 'black'.
#' @param alpha (dbl) Opacity. Default is 0.6.
#' @param prop (logical) make a stacked barplot a proportional barplot. Default is FALSE.
#' @param add_count_lab (logical) add count labels to bars. Default is TRUE.
#' @param vars_as_entered (logical) leave the variable order as entered, not flipping the values so the variable with more levels makes the bars. Default is FALSE.
#' @param legend_position (chr) position of the legend.
#'  Options Include: (default) 'top', 'bottom', 'left', 'right'
#' @param font_size (dbl) size of the font. Default is 3.25.
#'
#' @returns The ggplot object of the completed plot
#'
#' @examples
#' \dontrun{
#' generate_barplot(mtcars, "gear")
#'
#' generate_barplot(mtcars, "gear",
#'   groupby = "cyl", xlab = "gear", ylab = "Count",
#'   title = "Gear Vs. Carb", col_set = "Set2",
#'   prop = T, add_count_lab = T, vars_as_entered = F,
#'   legend_position = "top", font_size = 4
#' )
#' }
generate_barplot <- function(
    data, y = NULL, group = NULL, label_name = "labels",
    value_name = "values", groupby = "group", xlab = NULL, ylab = NULL, title = "Bar Plot",
    col_set = "Set1", cols = NULL, col_outline = "grey30", col_labels = "black", alpha = 0.6,
    prop = F, add_count_lab = T, vars_as_entered = F, legend_position = "top", font_size = 3.25) {
  suppressPackageStartupMessages(require(ggplot2))

  # Angled text threshold
  VERT_LEVELS_MIN <- 9 # minimum number of levels
  VERT_WORD_MIN <- 7 # minimum length of largest label name

  data <- convert_to_dataframe(data)
  label_name <- get_default_columns(data, label_name)
  data <- validate_data(data, label_name)

  if (!is.null(y)) {
    y <- convert_to_dataframe(y)
    value_name <- get_default_columns(y, value_name)
    # check if value_name is in y or add it if possible, otherwise error
    y <- validate_data(y, value_name)
    data <- concatenate_datasets(data, y, how = "horizontal")
  }

  if (!is.null(group)) {
    group <- convert_to_dataframe(group)
    groupby <- get_default_columns(group, groupby)
    group <- validate_data(group, groupby)
    data <- concatenate_datasets(data, group, how = "horizontal")
  }

  if (is.null(xlab)) {
    xlab <- label_name
  }

  # Check that the variable is a character
  if (!is.character(data[[label_name]])) {
    data[[label_name]] <- as.character(data[[label_name]])
  }
  cat_lvls <- length(unique(data[[label_name]]))

  if (!is.null(value_name) && value_name %in% colnames(data)) {
    data <- data[, c(label_name, value_name)]
    if (is.null(ylab)) {
      ylab <- value_name
    }
    sorted_inds <- order(data[[value_name]], decreasing = TRUE)
    data <- data[sorted_inds, ]
    max_length <- max(nchar(unique(na.omit(data[[label_name]]))))
    data[[label_name]] <- factor(data[[label_name]], levels = data[[label_name]])
    levels <- 1
    if (is.null(cols)) {
      cols <- color_select(levels, col_set)
    }
    the_plot <- ggplot(data, aes(x = .data[[label_name]], y = .data[[value_name]])) +
      geom_bar(stat = "identity", color = col_outline, fill = cols, alpha = alpha) +
      theme_bw() +
      labs(x = xlab, y = ylab, title = title) +
      theme(
        legend.position = legend_position,
        plot.title = element_text(hjust = 0.5)
      ) +
      {
        if (cat_lvls >= VERT_LEVELS_MIN || max_length >= VERT_WORD_MIN) {
          theme(axis.text.x = element_text(angle = 60, hjust = 1))
        }
      } +
      {
        if (add_count_lab) {
          geom_text(aes(label = sci_format(.data[[value_name]])),
            stat = "identity",
            position = position_stack(vjust = 0.5), color = col_labels, size = font_size
          )
        }
      }
  } else {
    if (is.null(ylab)) {
      ylab <- "Count"
    }
    # If a groupby value has been passed, the function sets up for a stacked
    # barplot
    if (!is.null(groupby) && groupby %in% colnames(data)) {
      stacked <- TRUE

      # Check that the variable is a character
      if (!is.character(data[[groupby]])) {
        data[[groupby]] <- as.character(data[[groupby]])
      }

      # settings for plot type and labels Proportional or Stack
      if (prop) {
        position_set <- "fill"
        position_func <- position_fill(vjust = 0.5)

        # Won't change the name if a custom label is given
        if (ylab == "Count") {
          ylab <- "Proportion"
        }
      } else {
        position_set <- "stack"
        position_func <- position_stack(vjust = 0.5)
      }

      # Calculate Levels
      groupby_lvls <- length(unique(data[[groupby]]))
      max_length <- max(nchar(unique(na.omit(data[[label_name]]))))

      # Function will make the variable with more levels the bars of the plot
      # Unless it is specified to leave it as is
      if (!vars_as_entered) {
        # if grouby is larger, swap the values
        if (groupby_lvls > cat_lvls) {
          temp <- label_name
          label_name <- groupby
          groupby <- temp
          xlab <- label_name

          levels <- cat_lvls
          cat_lvls <- groupby_lvls
          max_length <- max(nchar(unique(na.omit(data[[label_name]]))))
        } else {
          levels <- groupby_lvls
        }
      } else {
        levels <- groupby_lvls
      }
    } else {
      stacked <- FALSE
      cat_lvls <- length(unique(data[[label_name]]))
      max_length <- max(nchar(unique(na.omit(data[[label_name]]))))
      levels <- 1
    }

    if (is.null(cols)) {
      cols <- color_select(levels, col_set)
    }

    if (stacked) {
      the_plot <- ggplot(data, aes(
        x = forcats::fct_infreq(.data[[label_name]]),
        fill = forcats::fct_rev(forcats::fct_infreq(.data[[groupby]]))
      )) +
        geom_bar(position = position_set, stat = "count", color = col_outline, alpha = alpha) +
        theme_bw() +
        labs(
          x = xlab, y = ylab, color = groupby, fill = groupby,
          title = title
        ) +
        theme(legend.position = legend_position, plot.title = element_text(hjust = 0.5)) +
        scale_fill_manual(values = cols) +
        {
          if (cat_lvls >= VERT_LEVELS_MIN || max_length >= VERT_WORD_MIN) {
            theme(axis.text.x = element_text(angle = 60, hjust = 1))
          }
        } +
        {
          if (add_count_lab) {
            geom_text(aes(label = after_stat(count)),
              stat = "count", position = position_func,
              color = col_labels, size = font_size
            )
          }
        }
    } else {
      the_plot <- ggplot(data, aes(x = forcats::fct_infreq(.data[[label_name]]))) +
        geom_bar(stat = "count", color = col_outline, alpha = alpha) +
        theme_bw() +
        labs(x = xlab, y = ylab, title = title) +
        theme(
          legend.position = legend_position,
          plot.title = element_text(hjust = 0.5)
        ) +
        {
          if (cat_lvls >= VERT_LEVELS_MIN || max_length >= VERT_WORD_MIN) {
            theme(axis.text.x = element_text(angle = 60, hjust = 1))
          }
        } +
        {
          if (add_count_lab) {
            geom_text(aes(label = after_stat(count)),
              stat = "count", position = position_stack(vjust = 0.5),
              color = col_labels, size = font_size
            )
          }
        }
    }
  }
  return(the_plot)
}

#'
#' `generate_boxplot()` creates a violin plot using ggplot2
#'
#' Note: Requires ggplot2
#'
#' @param data (data.frame or vector) data frame containing the variable or numerical data
#' @param labels (data.frame or vector) vector (or data frame) containing the categorical data
#' @param column (chr) name of numerical variable column.
#' @param label_name (chr) name of categorical variable.
#' @param xlab (chr) label for categorical axis.
#' @param ylab (chr) label for numerical axis.
#'  Note: You don't need to change the x and y labels around if horizontal_plot is TRUE, the function will change them automatically.
#' @param title (chr) title of the plot. Default is 'Violin Plot'
#' @param legend_position (chr) position of the legend.
#'  Options Include: (default) 'top', 'bottom', 'left', 'right'
#' @param add_box (logical) Add boxplots on top of violin plots. Default is TRUE
#' @param horizontal_plot (logical) if you want the boxplots to be running horizontally, set to TRUE. Default is FALSE.
#' @param order (logical) if you want the violin plots ordered by median. Default is FALSE.
#' @param col_set (chr) name of RColorBrewer set to be used for colouring.
#'  Options Include: (default) 'Set1', 'Set2', 'Set3', 'Pastel2', 'Pastel1', 'Paired', 'Dark2', 'Accent'
#' @param cols (chr (vec)) vector of colors that you wish to use. Default to NULL and other colours are produced using the set in col_set.
#' @param alpha (dbl) Opacity. Default is 0.6.
#' @param log_num (chr) logarithmic transformation of the numerical axis (y).
#'   Options include: (default) NULL (none), 'log', 'log2', 'log10', 'log1p', 'log2_1p', 'log10_1p'.
#'
#' @returns The ggplot object of the completed plot
#'
#' @examples
#' \dontrun{
#' generate_boxplot(iris, column = "Petal.Length", label_name = "Species")
#'
#' generate_boxplot(iris,
#'   column = "Petal.Length", label_name = "Species",
#'   ylab = "Petal Length", xlab = "Species", horizontal_plot = T
#' )
#' }
generate_boxplot <- function(
    data, labels = NULL, column = NULL, label_name = "labels",
    xlab = NULL, ylab = NULL, title = "Boxplot", legend_position = "top", horizontal_plot = F,
    order = F, col_set = "Set1", cols = NULL, alpha = 0.6, log_num = NULL) {
  suppressPackageStartupMessages(require(ggplot2))

  # Angled text threshold
  VERT_LEVELS_MIN <- 9 # minimum number of levels
  VERT_WORD_MIN <- 7 # minimum length of largest label name

  data <- convert_to_dataframe(data)

  if (!is.null(labels)) {
    labels <- convert_to_dataframe(labels)
    label_name <- get_default_columns(labels, label_name)
    labels <- validate_data(labels, label_name)
  }

  if (is.null(column) && ncol(data) > 2 && !is.null(label_name)) {
    suppressPackageStartupMessages(require(reshape2))

    if (!is.null(labels)) {
      data <- concatenate_datasets(data, labels, how = "horizontal")
    }
    data <- reshape2::melt(data, id.vars = label_name)
    column <- names(data)[3]
  } else {
    column <- get_default_columns(data, column)
    data <- validate_data(data, column)
    if (!is.null(labels)) {
      data <- concatenate_datasets(data, labels, how = "horizontal")
    }
  }

  if (is.null(xlab)) {
    xlab <- label_name
  }

  if (is.null(ylab)) {
    ylab <- column
  }

  if (!is.null(log_num)) {
    log_num <- match.arg(log_num, c(
      "log2", "log10", "log", "log2_1p", "log10_1p",
      "log1p"
    ))
    ylab <- prepare_axis_label(ylab, log_num)
  }

  if (!is.null(label_name) && label_name %in% colnames(data)) {
    # Make sure label_name is categorical
    if (!is.character(data[[label_name]])) {
      data[[label_name]] <- as.character(data[[label_name]])
    }

    levels <- length(unique(data[[label_name]]))
    max_length <- max(nchar(unique(na.omit(data[[label_name]]))))

    if (is.null(cols)) {
      cols <- color_select(levels, col_set)
    }

    if (order) {
      the_plot <- ggplot(data, aes(x = forcats::fct_reorder(.data[[label_name]],
        .data[[column]],
        .fun = median
      ), y = .data[[column]], fill = .data[[label_name]])) +
        geom_boxplot(alpha = alpha) +
        theme_bw() +
        scale_fill_manual(values = cols) +
        labs(x = xlab, y = ylab, title = title) +
        theme(
          legend.position = legend_position,
          plot.title = element_text(hjust = 0.5)
        ) +
        {
          if (!is.null(log_num)) {
            scale_y_continuous(trans = log_num)
          }
        } +
        {
          if (horizontal_plot) {
            coord_flip()
          }
        } +
        {
          if ((levels >= VERT_LEVELS_MIN || max_length >= VERT_WORD_MIN)) {
            theme(axis.text.x = element_text(angle = 60, hjust = 1))
          }
        }
    } else {
      the_plot <- ggplot(data, aes(
        x = .data[[label_name]], y = .data[[column]],
        fill = .data[[label_name]]
      )) +
        geom_boxplot(alpha = alpha) +
        theme_bw() +
        scale_fill_manual(values = cols) +
        labs(x = xlab, y = ylab, title = title) +
        theme(
          legend.position = legend_position,
          plot.title = element_text(hjust = 0.5)
        ) +
        {
          if (!is.null(log_num)) {
            scale_y_continuous(trans = log_num)
          }
        } +
        {
          if (horizontal_plot) {
            coord_flip()
          }
        } +
        {
          if ((levels >= VERT_LEVELS_MIN || max_length >= VERT_WORD_MIN)) {
            theme(axis.text.x = element_text(angle = 60, hjust = 1))
          }
        }
    }
  } else {
    the_plot <- ggplot(data, aes(x = 1, y = .data[[column]])) +
      geom_boxplot(alpha = alpha) +
      theme_bw() +
      labs(x = xlab, y = ylab, title = title) +
      theme(
        legend.position = legend_position,
        plot.title = element_text(hjust = 0.5)
      ) +
      {
        if (!is.null(log_num)) {
          scale_y_continuous(trans = log_num)
        }
      } +
      {
        if (horizontal_plot) {
          coord_flip()
        }
      }
  }
  return(the_plot)
}

#'
#' `generate_violin()` creates a violin plot using ggplot2
#'
#' Note: Requires ggplot2
#'
#' @param data (data.frame or vector) data frame containing the variable or numerical data
#' @param labels (data.frame or vector) vector (or data frame) containing the categorical data
#' @param column (chr) name of numerical variable column.
#' @param label_name (chr) name of categorical variable.
#' @param xlab (chr) label for categorical axis.
#' @param ylab (chr) label for numerical axis.
#'  Note: You don't need to change the x and y labels around if horizontal_plot is TRUE, the function will change them automatically.
#' @param title (chr) title of the plot. Default is 'Violin Plot'
#' @param legend_position (chr) position of the legend.
#'  Options Include: (default) 'top', 'bottom', 'left', 'right'
#' @param add_box (logical) Add boxplots on top of violin plots. Default is TRUE
#' @param horizontal_plot (logical) if you want the boxplots to be running horizontally, set to TRUE. Default is FALSE.
#' @param order (logical) if you want the violin plots ordered by median. Default is FALSE.
#' @param col_set (chr) name of RColorBrewer set to be used for colouring.
#'  Options Include: (default) 'Set1', 'Set2', 'Set3', 'Pastel2', 'Pastel1', 'Paired', 'Dark2', 'Accent'
#' @param cols (chr (vec)) vector of colors that you wish to use. Default to NULL and other colours are produced using the set in col_set.
#' @param alpha (dbl) Opacity. Default is 0.6.
#' @param log_num (chr) logarithmic transformation of the numerical axis (y).
#'   Options include: (default) NULL (none), 'log', 'log2', 'log10', 'log1p', 'log2_1p', 'log10_1p'.
#'
#' @returns The ggplot object of the completed plot
#'
#' @examples
#' \dontrun{
#' generate_violin(iris, column = "Petal.Length", label_name = "Species")
#'
#' generate_violin(iris,
#'   column = "Petal.Length", label_name = "Species",
#'   ylab = "Petal Length", xlab = "Species", horizontal_plot = T
#' )
#' }
generate_violin <- function(
    data, labels = NULL, column = NULL, label_name = "labels",
    ylab = NULL, xlab = NULL, ylim = NULL, title = "Violin Plot",
    legend_position = "top", add_box = TRUE, horizontal_plot = FALSE,
    order = FALSE, col_set = "Set1", cols = NULL, alpha = 0.6,
    log_num = NULL, show_outliers = TRUE) {
  suppressPackageStartupMessages(require(ggplot2))

  # Angled text threshold
  VERT_LEVELS_MIN <- 9 # minimum number of levels
  VERT_WORD_MIN <- 7 # minimum length of largest label name

  data <- convert_to_dataframe(data)

  if (!is.null(labels)) {
    labels <- convert_to_dataframe(labels)
    label_name <- get_default_columns(labels, label_name)
    labels <- validate_data(labels, label_name)
  }

  if (is.null(column) && ncol(data) > 2 && !is.null(label_name)) {
    suppressPackageStartupMessages(require(reshape2))
    # melt the data
    if (!is.null(labels)) {
      data <- concatenate_datasets(data, labels, how = "horizontal")
    }
    data <- reshape2::melt(data, id.vars = label_name)
    column <- names(data)[3]
  } else {
    column <- get_default_columns(data, column)
    data <- validate_data(data, column)
    if (!is.null(labels)) {
      data <- concatenate_datasets(data, labels, how = "horizontal")
    }
  }


  if (is.null(xlab) && !is.null(label_name) && label_name %in% colnames(data)) {
    xlab <- label_name
  }

  if (is.null(ylab)) {
    ylab <- column
  }

  if (!is.null(log_num)) {
    log_num <- match.arg(log_num, c(
      "log2", "log10", "log", "log2_1p", "log10_1p",
      "log1p"
    ))
    # ylab <- prepare_axis_label(ylab, log_num)
    data[[column]] <- log_transformation(data[[column]], log_num)
  }

  if (!is.null(label_name) && label_name %in% colnames(data)) {
    # Make sure label_column is categorical
    if (!is.character(data[[label_name]])) {
      data[[label_name]] <- as.character(data[[label_name]])
    }

    levels <- length(unique(data[[label_name]]))
    max_length <- max(nchar(unique(na.omit(data[[label_name]]))))

    if (is.null(cols)) {
      cols <- color_select(levels, col_set)
    }

    if (order) {
      the_plot <- ggplot(data, aes(x = forcats::fct_reorder(.data[[label_name]],
        .data[[column]],
        .fun = median
      ), y = .data[[column]], fill = .data[[label_name]])) +
        geom_violin(alpha = alpha) +
        {
          if (add_box) {
            geom_boxplot(
              width = 0.2, color = "black", outlier.shape = 1,
              fill = NA, outlier.fill = "black", alpha = alpha,
              outliers = show_outliers
            )
          }
        } +
        theme_bw() +
        scale_fill_manual(values = cols) +
        labs(
          x = xlab, y = ylab,
          title = title
        ) +
        theme(legend.position = legend_position, plot.title = element_text(hjust = 0.5)) +
        # {
        #   if (!is.null(log_num)) {
        #     scale_y_continuous(trans = log_num)
        #   }
        # } +
        {
          if (!is.null(ylim)) {
            ylim(ylim[[1]], ylim[[2]])
          }
        } +
        {
          if (horizontal_plot) {
            coord_flip()
          }
        } +
        {
          if ((levels >= VERT_LEVELS_MIN || max_length >= VERT_WORD_MIN)) {
            theme(axis.text.x = element_text(angle = 60, hjust = 1))
          }
        }
    } else {
      the_plot <- ggplot(data, aes(
        x = .data[[label_name]], y = .data[[column]],
        fill = .data[[label_name]]
      )) +
        geom_violin(alpha = alpha) +
        {
          if (add_box) {
            geom_boxplot(
              width = 0.7, color = "black", outlier.shape = 1, outlier.size = 0.5,
              fill = NA, outlier.fill = "black", alpha = alpha, outliers = show_outliers
            )
          }
        } +
        theme_bw() +
        scale_fill_manual(values = cols) +
        labs(
          x = xlab, y = ylab,
          title = title
        ) +
        theme(legend.position = legend_position, plot.title = element_text(hjust = 0.5)) +
        # {
        #   if (!is.null(log_num)) {
        #     scale_y_continuous(trans = log_num)
        #   }
        # } +
        {
          if (!is.null(ylim)) {
            ylim(ylim[[1]], ylim[[2]])
          }
        } +
        {
          if (horizontal_plot) {
            coord_flip()
          }
        } +
        {
          if ((levels >= VERT_LEVELS_MIN || max_length >= VERT_WORD_MIN)) {
            theme(axis.text.x = element_text(angle = 60, hjust = 1))
          }
        }
    }
  } else {
    # generate a single violin plot
    the_plot <- ggplot(data, aes(x = 1, y = .data[[column]])) +
      geom_violin(fill = "grey40", alpha = alpha) +
      {
        if (add_box) {
          geom_boxplot(
            width = 0.2, color = "black", outlier.shape = 1,
            fill = NA, outlier.fill = "black", alpha = alpha
          )
        }
      } +
      theme_bw() +
      labs(x = xlab, y = ylab, title = title) +
      theme(
        legend.position = legend_position,
        plot.title = element_text(hjust = 0.5)
      ) +
      {
        if (!is.null(log_num)) {
          scale_y_continuous(trans = log_num)
        }
      } +
      {
        if (horizontal_plot) {
          coord_flip()
        }
      }
  }
  return(the_plot)
}

#'
#' `generate_scatterplot()` creates a scatterplot using ggplot2
#'
#' Note: Requires ggplot2 package
#'
#' @param data (data.frame or vector) data frame containing the variables or data for the x axis.
#' @param y (data.frame or vector) data for the y axis.
#' @param group (data.frame or vector) label data for grouping the x and y by colour.
#' @param xdata (chr) name of column with data for the x-axis.
#' @param ydata (chr) name of column with data for the y-axis.
#' @param groupby (chr) name of categorical variable to group the points. Default is NULL.
#' @param xlab (chr) x-axis label. Default is 'Var 1'.
#' @param ylab (chr) y-axis label. Default is 'Var 2'.
#' @param title (chr) title for the plot. Default is 'Scatterplot'.
#' @param alpha (dbl) opacity of points. Range: 0 to 1. Default is 1.
#' @param col_set (chr) name of RColorBrewer set to be used for colouring.
#'  Options Include: (default) 'Set1', 'Set2', 'Set3', 'Pastel2', 'Pastel1', 'Paired', 'Dark2', 'Accent'
#' @param xlog (chr) log transformation of the x-axis.
#'  Options Include: (default) NULL (None), 'log', 'log2', 'log10', 'log1p', 'log2_1p', 'log10_1p'.
#' @param ylog (chr) log transformation of the y-axis.
#'  Options Include: (default) NULL (None), 'log', 'log2', 'log10', 'log1p', 'log2_1p', 'log10_1p'.
#'
#' @returns The ggplot object of the completed plot
#'
#' @examples
#' \dontrun{
#' generate_scatterplot(mtcars, xdata = "wt", ydata = "qsec")
#'
#' generate_scatterplot(iris,
#'   xdata = "Petal.Length", ydata = "Petal.Width",
#'   groupby = "Species", xlab = "Petal Length",
#'   ylab = "Petal Width", title = "Petals (Length VS. Width)",
#'   alpha = 0.75, col_set = "Set2"
#' )
#' }
generate_scatterplot <- function(
    data, y = NULL, group = NULL, xdata = "x", ydata = "y",
    groupby = "group", xlab = NULL, ylab = NULL, title = "Scatterplot", alpha = 1,
    col_set = "Set1", cols = NULL, xlog = NULL, ylog = NULL) {
  suppressPackageStartupMessages(require(ggplot2))
  data <- convert_to_dataframe(data)
  xdata <- get_default_columns(data, xdata)
  data <- validate_data(data, xdata)

  if (!is.null(y)) {
    y <- convert_to_dataframe(y)
    value_name <- get_default_columns(y, ydata)
    y <- validate_data(y, ydata)
    data <- concatenate_datasets(data, y, how = "horizontal")
  } else {
    data <- validate_data(data, ydata)
  }

  if (!is.null(group)) {
    group <- convert_to_dataframe(group)
    groupby <- get_default_columns(group, groupby)
    group <- validate_data(group, groupby)
    data <- concatenate_datasets(data, group, how = "horizontal")
  }

  if (is.null(xlab)) {
    xlab <- xdata
  }

  if (is.null(ylab)) {
    ylab <- ydata
  }

  if (!is.null(xlog)) {
    xlog <- match.arg(xlog, c(
      "log2", "log10", "log", "log2_1p", "log10_1p",
      "log1p"
    ))
    xlab <- prepare_axis_label(xlab, xlog)
  }
  if (!is.null(ylog)) {
    ylog <- match.arg(ylog, c(
      "log2", "log10", "log", "log2_1p", "log10_1p",
      "log1p"
    ))
    ylab <- prepare_axis_label(ylab, ylog)
  }

  if (!is.null(groupby) && groupby %in% colnames(data)) {
    grouped <- TRUE

    if (!is.character(data[[groupby]])) {
      data[[groupby]] <- as.character(data[[groupby]])
    }

    if (is.null(cols)) {
      levels <- length(unique(data[[groupby]]))
      cols <- color_select(levels, col_set)
    }
  } else {
    grouped <- FALSE
  }

  if (grouped) {
    ggplot(data, aes(x = .data[[xdata]], y = .data[[ydata]], col = .data[[groupby]])) +
      geom_point(alpha = alpha) +
      theme_bw() +
      theme(plot.title = element_text(hjust = 0.5)) +
      scale_color_manual(values = cols) +
      labs(x = xlab, y = ylab, title = title) +
      {
        if (!is.null(xlog)) {
          scale_x_continuous(trans = xlog)
        }
      } +
      {
        if (!is.null(ylog)) {
          scale_y_continuous(trans = ylog)
        }
      }
  } else {
    ggplot(data, aes(x = .data[[xdata]], y = .data[[ydata]])) +
      geom_point(alpha = alpha) +
      theme_bw() +
      theme(plot.title = element_text(hjust = 0.5)) +
      labs(
        x = xlab,
        y = ylab, title = title
      ) +
      {
        if (!is.null(xlog)) {
          scale_x_continuous(trans = xlog)
        }
      } +
      {
        if (!is.null(ylog)) {
          scale_y_continuous(trans = ylog)
        }
      }
  }
}

#' calls all the plots for a single categorical variable
#' @param data (data.frame) data frame containing the data.
#' @param label_name (chr) name of categorical variable.
plot_cat_metadata <- function(data, label_name, title = NULL, ...) {
  if (is.null(title)) {
    title <- paste0("Distribution of ", label_name)
  }
  p <- generate_barplot(data,
    label_name = label_name, xlab = label_name, title = title,
    ...
  )
  return(p)
}

#' calls all the plots for a single numerical variable
#' @param data (data.frame) data frame containing the data.
#' @param num_var (chr) name of numerical variable.
plot_num_metadata <- function(data, num_var, title = NULL, ...) {
  if (is.null(title)) {
    title <- paste0("Distribution of ", num_var)
  }
  p <- generate_histogram(data, num_var, xlab = num_var, title = title, ...)
  return(p)
}

#' Categorical Variable vs Categorical Variable
#' @param data (data.frame) data frame containing the data.
#' @param var1 (chr) name of 1st categorical variable.
#' @param var2 (chr) name of 2nd categorical variable.
plot_cat_vs_cat_metadata <- function(data, var1, var2, title = NULL, ...) {
  suppressPackageStartupMessages(require(patchwork))
  # Categorical vs Categorical
  if (is.null(title)) {
    title <- paste0(var1, " VS. ", var2)
  }
  # Stacked Barplot
  p1 <- generate_barplot(data,
    label_name = var1, groupby = var2, xlab = var1,
    title = NULL, add_count_lab = F, ...
  )
  # Proportional Barplot
  p2 <- generate_barplot(data,
    label_name = var1, xlab = var1, groupby = var2,
    prop = T, title = NULL, ...
  )
  # Grid
  p <- (p1 + p2 + plot_layout(guides = "collect") + plot_annotation(
    title = title,
    theme = theme(plot.title = element_text(hjust = 0.5))
  ) & theme(legend.position = "top"))
  return(p)
}

#' Categorical Variable vs Numerical Variable
#' @param data (data.frame) data frame containing the data.
#' @param cat (chr) name of categorical variable.
#' @param num (chr) name of numerical variable.
plot_cat_vs_num_metadata <- function(data, cat, num, title = NULL, ...) {
  suppressPackageStartupMessages(require(patchwork))
  # Categorical vs Numeric
  if (is.null(title)) {
    title <- paste0(cat, " VS. ", num)
  }
  # Violin Plot
  p <- generate_violin(data, column = num, label_name = cat, title = title, ...)
  return(p)
}

#' Numerical Variable vs Numerical Variable
#' @param data (data.frame) data frame containing the data.
#' @param var1 (chr) name of 1st numerical variable.
#' @param var2 (chr) name of 2nd numerical variable.
plot_num_vs_num_metadata <- function(data, var1, var2, title = NULL, ...) {
  # Numeric vs Numeric
  if (is.null(title)) {
    title <- paste0(var1, " VS. ", var2)
  }
  # Scatterplot
  p <- generate_scatterplot(data,
    xdata = var1, ydata = var2, xlab = var1, ylab = var2,
    title = title, ...
  )
  return(p)
}
