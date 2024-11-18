source(file.path(R_SCRIPTS_PATH, "plotting_utils.R"))
source(file.path(R_SCRIPTS_PATH, "utils.R"))

plot_sample_metadata <- function(
    data, outcome = NULL,
    sample_metadata_columns = NULL,
    outcome_column = NULL,
    path,
    device = "pdf") {

  suppressPackageStartupMessages(require(ggplot2))
  suppressPackageStartupMessages(require(RColorBrewer))
  suppressPackageStartupMessages(require(circlize))
  suppressPackageStartupMessages(require(patchwork))
  width <- 7
  height <- 7
  if (device[1] != ".") {
    device <- paste0(".", device)
  }
  data <- convert_to_dataframe(data)
  sample_metadata_columns <- get_default_columns(
    data, sample_metadata_columns,
    max_cols = NULL
  )
  data <- validate_data(data, sample_metadata_columns)

  if (!is.null(outcome)) {
    outcome <- convert_to_dataframe(outcome)
    outcome_column <- get_default_columns(outcome, outcome_column, max_cols = 1)
    outcome <- validate_data(outcome, outcome_column)
    data <- concatenate_datasets(
      data[, sample_metadata_columns], outcome,
      how = "horizontal"
    )
  } else if (!is.null(outcome_column)) {
    outcome_column <- get_default_columns(data, outcome_column, max_cols = 1)
  } else {
    stop("Please provide either an outcome or an outcome column")
  }

  # get filename from path
  file_name <- gsub("\\.[^.]*$", "", basename(path))
  path <- dirname(path)
  # Get Outcome Type
  outcome_type <- detect_var_type(data, outcome_column)

  metadata_dir <- path
  if (!dir.exists(metadata_dir)) {
    dir.create(metadata_dir)
  }

  # Comparison Visualizations
  if (outcome_type == "categorical") {
    suppressPackageStartupMessages(require(forcats))
    # Outcome variable visualization
    cat_metadata <- plot_cat_metadata(data, outcome_column)
    cat_metadata_fn <- paste0(file_name, "_dist_", outcome_column, device)
    save_plots(cat_metadata_fn,
      plot = cat_metadata,
      path = metadata_dir, width = width, height = height
    )


    plot_num <- 0
    # Loop through all the columns
    for (col in names(data)) {
      plot_num <- plot_num + 1
      # If the col is the same as the outcome
      if (col == outcome_column) next

      # Get the data type of the current column
      col_type <- detect_var_type(data, col)

      # If it is other, we skip for now
      if (col_type == "other") {
        print(paste0(col, " was not plotted"))
        next
      }

      # categorical and numerical comparisons
      if (col_type == "categorical") {
        comp_metadata <- plot_cat_vs_cat_metadata(data, outcome_column, col)
      } else if (col_type == "numerical") {
        comp_metadata <- plot_cat_vs_num_metadata(data, outcome_column, col)
      }

      # Save Comparison
      comp_metadata_fn <- paste0(
        file_name, plot_num, "_",
        outcome_column, "_vs_", col, device
      )
      save_plots(comp_metadata_fn,
        plot = comp_metadata,
        path = metadata_dir, width = width * 2, height = height
      )
    }
  } else if (outcome_type == "numerical") {
    # Outcome variable visualization
    num_metadata <- plot_num_metadata(data, outcome_column, font_size = 11)
    num_metadata_fn <- paste0(file_name, "_dist_", outcome_column, device)
    save_plots(num_metadata_fn,
      plot = num_metadata, path = metadata_dir,
      width = width, height = height
    )
    plot_num <- 0
    # Loop through all of the columns
    for (col in names(data)) {
      plot_num <- plot_num + 1
      # If the col is the same as the outcome
      if (col == outcome_column) next

      # Get the data type of the current column
      col_type <- detect_var_type(data, col)

      # If it is other, we skip for now
      if (col_type == "other") {
        print(paste0(col, " was not plotted"))
        next
      }

      # categorical and numerical comparisons
      if (col_type == "categorical") {
        comp_metadata <- plot_cat_vs_num_metadata(data, col, outcome_column)
      } else if (col_type == "numerical") {
        comp_metadata <- plot_num_vs_num_metadata(data, outcome_column, col)
      }

      comp_metadata_fn <- paste0(
        file_name, plot_num, "_", outcome_column, "_vs_", col, device
      )
      save_plots(
        comp_metadata_fn,
        plot = comp_metadata,
        path = metadata_dir, width = width * 2, height = height
      )
    }
  } else if (outcome_type == "other") {
    print("Outcome variable has too many levels")
  }
}
