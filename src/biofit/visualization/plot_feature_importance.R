source(file.path(R_SCRIPTS_PATH, "plotting_utils.R"))
source(file.path(R_SCRIPTS_PATH, "utils.R"))

#'
#' Feature importance plot with feature data in the samples.
#' @param X Arrow data table that is logged or in non-logged form depending on the omics data
#' @param y Arrow labels
#' @param sample_metadata Arrow sample metadata
#' @param feature_importances Arrow feature importance table: first column is the feature ID, "feature", second column (and on wards) is/are the importance per final model seed(s)
#' @param path the path for the pdf file to save
#' @param X, y, sample_metadata, feature_importances arrow objects from the biofit framework
#' @param plot_top the number of top features to plot. Default is 15.
#' @param feature_meta_name can be NULL, prints the feature ID; or character such as "species", prints this column, or vector of more than 1 characters c("genus", "species"), then prints the collapsed text with a space in between. Default is NULL.
#' @param plot_title name of the data type, eg "Presence", or "log2 Abundance". Default is NULL.
#' @param column_title eg. "Sample" or "Isolate". Default is NULL.
#' @param row_title eg. "OTU" or "SNP". Default is NULL.
#' @examples
#' \dontrun{
#' # usage for OTU data:
#' plot_feature_importance(X, y, sample_metadata, feature_importances, plot_top = 30, feature_meta_name = c("feature", "genus", "species"), plot_title = "log2\nAbundance", column_title = "Sample", row_title = "OTU")
#' # usage for metagenomics data:
#' plot_feature_importance(X, y, sample_metadata, feature_importances, plot_top = 30, feature_meta_name = c("genus", "species"), plot_title = "log2\nAbundance", column_title = "Sample", row_title = "Taxonomy")
#' # usage for transcriptomics/proteomics (non-MALDI) data: eg
#' plot_feature_importance(X, y, sample_metadata, feature_importances, plot_top = 30, feature_meta_name = c("gene"), plot_title = "log2\nExpression", column_title = "Sample", row_title = "Gene")
#' # usage for genomics data:
#' plot_feature_importance(X, y, sample_metadata, feature_importances, plot_top = 30, feature_meta_name = NULL, plot_title = "Presence", column_title = "Isolate", row_title = "SNP")
#' # usage for MALDI-TOF data:
#' plot_feature_importance(X, y, sample_metadata, feature_importances, plot_top = 30, feature_meta_name = "DaRange", plot_title = "log10\nAbundance", column_title = "Isolate", row_title = "Da range")
#' }
#'
plot_feature_importance <- function(
    X,
    feature_importances,
    path,
    y = NULL,
    sample_metadata = NULL,
    feature_metadata = NULL,
    input_columns = NULL,
    target_columns = NULL,
    sample_metadata_columns = NULL,
    feature_column = NULL,
    feature_meta_name = NULL,
    plot_top = 15,
    cols = NULL,
    col_heat = NULL,
    show_column_names = FALSE,
    plot_title = "Relative Abundance",
    column_title = "Samples",
    row_title = "Taxonomic Relative Abundance") {
  width <- 13
  height <- 7

  X <- convert_to_dataframe(X)
  input_columns <- get_default_columns(X, input_columns, NULL)
  X <- validate_data(X, input_columns)

  if (!is.null(y)) {
    y <- convert_to_dataframe(y)
    target_columns <- get_default_columns(y, target_columns)
    y <- validate_data(y, target_columns)
  }

  if (!is.null(sample_metadata)) {
    sample_metadata <- convert_to_dataframe(sample_metadata)
    sample_metadata_columns <- get_default_columns(
      sample_metadata, sample_metadata_columns, NULL
    )
    sample_metadata <- validate_data(sample_metadata, sample_metadata_columns)
  }

  if (is.null(sample_column)) {
    sample_column <- sample_metadata_columns[1]
  }


  if (!is.null(feature_metadata)) {
    feature_metadata <- convert_to_dataframe(feature_metadata)
    if (is.list(feature_meta_name)) {
      feature_meta_name <- as.vector(unlist(feature_meta_name))
    }
    feature_meta_name <- get_default_columns(
      feature_metadata, feature_meta_name
    )
    feature_metadata <- validate_data(feature_metadata, feature_meta_name)
    rownames(feature_metadata) <- feature_metadata[, feature_column]
  }

  suppressPackageStartupMessages(require(ComplexHeatmap))
  suppressPackageStartupMessages(require(grid))
  suppressPackageStartupMessages(require(circlize))
  suppressPackageStartupMessages(require(RColorBrewer))

  ## get label and feature information
  ## convert arrow data format to data frame
  if (!is.null(feature_metadata)) {
    feature_metadata <- convert_to_dataframe(feature_metadata)
  }


  feature_importances <- convert_to_dataframe(feature_importances)
  top_feat_id <- feature_importances[, 1]
  rownames(feature_importances) <- feature_importances[, feature_column]
  # drop feature column
  not_feature_column <- setdiff(colnames(feature_importances), feature_column)
  feature_importances <- feature_importances[, not_feature_column]


  y <- factor(as.vector(convert_to_dataframe(y)[, 1]))

  rownames(X) <- sample_metadata[, sample_column]

  ### make sure plot_top is numeric
  if (!is.numeric(plot_top)) {
    stop("plot_top, the number of top features to plot, must be numeric.")
  }

  ### color selected from plotting_utils.R to be consistent
  if (is.null(cols)) {
    cols <- color_select(length(levels(y)))
    names(cols) <- levels(y)
  }

  ### if col_heat not specified
  if (is.null(col_heat)) {
    col_heat <- brewer.pal(9, "YlOrRd")
  }

  ## Define the feature/text annotation with row names
  ## plot feature ID if not specified
  if (is.null(feature_meta_name) || is.null(feature_metadata)) {
    feat_on_plot <- top_feat_id
  } else {
    ## for more than one feature meta columns to display
    if (length(feature_meta_name) > 1) {
      if (!(feature_column %in% feature_meta_name)) {
        feature_meta_name <- c(feature_column, feature_meta_name)
      }
      feat_on_plot <- apply(
        feature_metadata[top_feat_id, feature_meta_name],
        1,
        paste,
        collapse = "\n"
      )
    } else { ## for one feature meta columns to display
      feat_on_plot <- feature_metadata[top_feat_id, feature_meta_name]
    }
  }

  ##################
  ## generate feature importance plot
  ##################
  ### the feature boxplot/barplot
  # remove UNINTEGRATED from top_feat_id
  hdat2plot <- t(data.matrix(X[, top_feat_id]))

  print(hdat2plot)
  print(col_heat)
  if (ncol(feature_importances) == 1) {
    ha2 <- ComplexHeatmap::rowAnnotation(
      ` ` = row_anno_points(
        feature_importances[top_feat_id, ],
        axis = TRUE, outline = FALSE
      ),
      width = unit(3, "cm")
    )
  } else {
    ha2 <- ComplexHeatmap::rowAnnotation(
      ` ` = row_anno_boxplot(
        data.matrix(feature_importances[top_feat_id, ]),
        axis = TRUE,
        outline = FALSE
      ),
      width = unit(3, "cm")
    )
  }

  ### the sample annotation in the column
  ### (Added a name hack here to be Importance
  ### as the label is above feature importance box/dot plot)
  ha1 <- ComplexHeatmap::HeatmapAnnotation(
    df = data.frame(Importance = y),
    show_legend = TRUE,
    col = list(Importance = cols),
    annotation_legend_param = list(
      title = target_columns,
      legend_direction = "vertical"
    )
  )

  ### add feature names to rows
  text_annotation <- rowAnnotation(
    text = anno_text(
      feat_on_plot,
      just = "left",
      gp = gpar(fontsize = 10)
    )
  )

  ### plot histogram and add the 3 components from above
  h2plot <- ComplexHeatmap::Heatmap(hdat2plot,
    cluster_rows = FALSE, col = col_heat, top_annotation = ha1,
    heatmap_legend_param = list(
      title = plot_title, color_bar = "continuous",
      legend_direction = "vertical"
    ),
    show_column_names = show_column_names,
    row_title = row_title, # Title for the rows
    column_title = column_title, # Title for the columns
    row_title_side = "left", # Position of the row title
    column_title_side = "top"
  ) +
    ha2 + text_annotation # , top_annotation_height = unit(1, "cm")
  start_device(path, width = width, height = height, units = "in", res = 300)
  draw(h2plot)
  dev.off()
}
