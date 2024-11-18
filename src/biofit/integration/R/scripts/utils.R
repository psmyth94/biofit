BIOCONDUCTOR <- c(
  "limma", "metagenomeSeq", "phyloseq",
  "DESeq2", "biomaRt", "GenomicRanges",
  "Biostrings", "BSgenome", "GenomicFeatures",
  "GenomicAlignments", "VariantAnnotation",
  "S4Vectors", "IRanges", "AnnotationDbi",
  "ComplexHeatmap"
)

save_plots <- function(
    filename, plot = last_plot(), device = NULL, path = NULL,
    scale = 1, width = NA, height = NA, units = c(
      "in", "cm",
      "mm", "px"
    ), dpi = 300, limitsize = TRUE, bg = NULL,
    create.dir = TRUE, ...) {
  suppressPackageStartupMessages(c(
    require(ggplot2)
  ))


  ext <- NULL
  if (!is.null(filename)) {
    ext <- tolower(tools::file_ext(filename))
  } else if (!is.null(path)) {
    ext <- tolower(tools::file_ext(path))
  }

  # always save a png if ext is not png
  if (ext != "png") {
    fn <- paste0(tools::file_path_sans_ext(filename), ".png")

    ggplot2::ggsave(
      fn,
      plot = plot, device = "png",
      path = path, scale = scale, width = width,
      height = height, units = units, dpi = dpi,
      limitsize = limitsize, bg = bg, create.dir = create.dir, ...
    )
  }
  ggplot2::ggsave(
    filename,
    plot = plot, device = device,
    path = path, scale = scale, width = width,
    height = height, units = units, dpi = dpi,
    limitsize = limitsize, bg = bg, create.dir = create.dir, ...
  )
}

get_info_from_arrow_table <- function(arrow_table) {
  # Get the schema
  info <- arrow_table$schema$metadata$huggingface
  if (is.null(info)) {
    return(NULL)
  }
  jsonlite::fromJSON(info)$info$features
}

#' @title arrow_to_factor
#' @description Convert label encodings to factors.
#' Uses the arrow metadata to get the label information.
#' @param y The arrow table
#' @return if y is not an arrow table or y is not a class label,
#' it returns y as is.
encodings_to_factor <- function(y) {
  # verify that y is an arrow table
  if (!inherits(y, "Table")) {
    return(y)
  }
  label_info <- get_info_from_arrow_table(y)$features$labels
  if (is_class(label_info)) {
    return(y)
  }
  # Add a label for missing values
  label_names <- c("", label_info$names)
  y <- as.data.frame(y)[, 1]
  # labels ranges from -1 to n-1,
  # where n is the number of classes and a -1 is a missing value
  factor(y + 2,
    levels = seq_along(label_names),
    labels = label_names
  )
}

is_class <- function(label_info) {
  if (label_info$`_type` != "ClassLabel") {
    return(TRUE)
  }
  FALSE
}

#' detect using class what each variable type is
#' @param data data frame containing the data
#' @param col name of the column that is being checked
detect_var_type <- function(data, col) {
  cat_types <- c("character", "logical")
  num_types <- c("numeric", "integer", "double")

  if (class(data[[col]]) %in% cat_types) {
    col_type <- "categorical"

    if (is_other_var(data, col)) {
      col_type <- "other"
    }
  } else if (class(data[[col]]) %in% num_types) {
    col_type <- "numerical"

    if (!is_other_var(data, col)) {
      col_type <- "categorical"
    }
    if (length(unique(data[[col]])) <= 1) {
      col_type <- "other"
    }
  }
  return(col_type)
}

#' For detecting our classification of other variable
#' @param data data frame containing the data
#' @param col name of variable being checked
#' @param threshold max number of levels
is_other_var <- function(data, col, threshold = 30) {
  other_var <- FALSE

  num_levels <- length(unique(data[[col]]))
  # if there are more than 30 levels we consider this an other variable
  if (num_levels > threshold || num_levels <= 1) {
    other_var <- TRUE
  }

  return(other_var)
}

get_feature_metadata <- function(data) {
  # Create a metadata table for the features
  feature_metadata <- get_info_from_arrow_table(data)

  extract_metadata <- function(item) {
    return(item$metadata)
  }
  # Extract and combine the metadata from each item into a data frame
  metadata_list <- lapply(feature_metadata, extract_metadata)
  metadata_df <- dplyr::bind_rows(metadata_list)
  # add the feature names to the metadata
  metadata_df$feature <- colnames(data)
  return(metadata_df)
}

start_device <- function(path, ...) {
  # Start the device
  if (is.null(path)) {
    path <- tempfile()
  }
  ext <- tolower(tools::file_ext(path))
  if (ext == "pdf") {
    args <- list(...)
    pdf(path, width = args$width, height = args$height)
  } else if (ext == "png") {
    png(path, ...)
  } else if (ext == "jpeg" || ext == "jpg") {
    jpeg(path, ...)
  } else if (ext == "bmp") {
    bmp(path, ...)
  } else if (ext == "tiff") {
    tiff(path, ...)
  } else {
    stop("Unsupported file format")
  }
}


is_dataframe <- function(data) {
  is(data, "Table") ||
    is(data, "RecordBatch") ||
    is.data.frame(data)
}


is_vector <- function(data) {
  is(data, "Array") ||
    is(data, "ChunkedArray") ||
    is.vector(data)
}


convert_to_dataframe <- function(value) {
  if (is_dataframe(value)) {
    as.data.frame(value)
  } else if (is_vector(value)) {
    `colnames<-`(as.data.frame(value), NULL)
  } else {
    stop(paste0("Unsupported object type: ", class(data)))
  }
}

get_default_columns <- function(data, default = NULL, max_cols = 1) {
  if (
    is_dataframe(data) &&
      (
        is.null(max_cols) ||
          ncol(data) == max_cols
      ) &&
      !is.null(colnames(data))
  ) {
    colnames(data)
  } else {
    default
  }
}

validate_data <- function(data, names, required = FALSE, replace = TRUE) {
  if (is_dataframe(data)) {
    if (!is.null(names)) {
      if (!all(names %in% colnames(data)) && !replace) {
        stop(paste0("Column name ", names, " is not in the data"))
      }
    } else if (required) {
      stop("A column name was not provided")
    }
    if (ncol(data) == length(names) && replace) {
      colnames(data) <- names
    }
    data
  } else {
    stop("Data must be a data frame")
  }
}


concatenate_datasets <- function(data1, data2, how = "horizontal") {
  if (is_dataframe(data1) || is_dataframe(data2)) {
    if (how == "vertical") {
      rbind(data1, data2)
    } else {
      cbind(data1, data2)
    }
  } else if (is_vector(data1) && is_vector(data2)) {
    if (how == "vertical") {
      c(data1, data2)
    } else {
      cbind(data1, data2)
    }
  } else {
    stop("Data types must match")
  }
}
