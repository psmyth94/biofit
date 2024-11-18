source(paste0(R_SCRIPTS_PATH, "/utils.R"))

install_dependencies <- function() {
  dependencies <- c(
    "ggplot2", "arrow", "circlize", "RColorBrewer", "scales",
    "forcats", "patchwork", "reshape2", "ComplexHeatmap",
    "edgeR", "dplyr", "tools"
  )
  ensure_packages(dependencies)
}
