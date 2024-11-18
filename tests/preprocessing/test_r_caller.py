import os

import biofit.config
from biofit.integration.R.r_caller import RCaller
from tests.utils import require_rpy2


@require_rpy2
def test_r_caller(count_data):
    X, y = count_data
    otu_dataset = X
    X = otu_dataset.data.table

    os.makedirs(os.path.join(biofit.config.BIOFIT_CACHE_HOME, "outputs"), exist_ok=True)
    output_path = os.path.join(biofit.config.BIOFIT_CACHE_HOME, "outputs/histogram.png")

    r = RCaller.from_script(
        """
        plot_histogram <- function(X, output_path, breaks=30) {
            X <- as.data.frame(X)
            library(ggplot2)
            df <- data.frame(abundance=rowSums(X))
            p <- ggplot(df, aes(x=abundance)) +
                geom_histogram(bins=breaks) +
                theme_minimal()
        }
        """
    )
    r.verify_r_dependencies(bioconductor_dependencies=["edgeR"], install_missing=True)
    func = r.get_method("plot_histogram", exit_code="ggsave(output_path, plot=results)")

    func(X, output_path)


@require_rpy2
def test_r_caller_create_dataframe(count_data):
    X, y = count_data
    otu_dataset = X
    X = otu_dataset.data.table

    r = RCaller.from_script(
        """
        create_dataframe <- function(X) {
            X <- as.data.frame(X)
            return(X)
        }
        """
    )
    func = r.get_method("create_dataframe")

    func(X)


@require_rpy2
def test_r_caller_create_otu_dataframe(count_data):
    X, y = count_data
    otu_dataset = X
    X = otu_dataset.data.table

    r = RCaller.from_script(
        """
        create_otu_dataframe <- function(X) {
            X <- as.data.frame(X)
            df <- data.frame(abundance=rowSums(X))
        }
        """
    )
    func = r.get_method("create_otu_dataframe")

    func(X)
